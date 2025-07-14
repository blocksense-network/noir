use crate::{
    MonomorphizationRequest, State,
    ast::{
        BinaryOp, ExprF, Literal, SpannedExpr, SpannedOptionallyTypedExpr, SpannedTypedExpr,
        UnaryOp, cata, try_cata, try_contextual_cata,
    },
};
use noirc_frontend::{
    ast::IntegerBitSize,
    monomorphization::{FUNC_RETURN_VAR_NAME, ast::Type as NoirType},
    shared::Signedness,
};
use num_bigint::{BigInt, BigUint};
use num_traits::{One, Zero};

#[derive(Debug, Clone)]
pub enum TypeInferenceError {
    MonomorphizationRequest(MonomorphizationRequest),
    TypeError { got: Option<NoirType>, wanted: Option<NoirType>, message: Option<String> },
}

#[derive(Debug, PartialEq, Eq)]
enum FitsIn {
    Yes,
    No {
        /// `None` means there is no possible `IntegerBitSize` that could contain us
        need: Option<NoirType>,
    },
}
fn bi_can_fit_in(bi: &BigInt, hole_size: &IntegerBitSize, hole_sign: &Signedness) -> FitsIn {
    let is_negative = match bi.sign() {
        num_bigint::Sign::Minus => true,
        num_bigint::Sign::Plus | num_bigint::Sign::NoSign => false,
    };

    // NOTE: if we have a negative literal, but try to fit into an unsigned integer
    let is_wrong_sign = is_negative && !hole_sign.is_signed();

    let mut bits = bi.bits();

    // NOTE: Add one bit for the sign.
    if hole_sign.is_signed() {
        bits += 1;
    }

    if is_negative {
        fn is_power_of_two(n: &BigUint) -> bool {
            // 10000000
            // 01111111
            !n.is_zero() && ((n & (n - BigUint::one())) == BigUint::zero())
        }

        // NOTE: ...unless we have a negative number whose magnitude is a power of two.
        //       This is the because it fits perfectly into the two's complement minimum value.
        //       (e.g., -128 fits in `i8`, the same as -127, but not 128).
        if is_power_of_two(bi.magnitude()) {
            bits -= 1;
        }
    }

    // NOTE: find the smallest `IntegerBitSize` that could contain us, `None` if no such one exists
    let smallest_fit_size =
        IntegerBitSize::allowed_sizes().into_iter().find(|size| bits <= size.bit_size() as u64);

    // NOTE: using the `PartialOrd` instance for `IntegerBitSize`,
    //       which definitionally orders the bit sizes in an increasing order
    let size_fits = smallest_fit_size.map(|sfs| sfs <= *hole_size).unwrap_or(false);

    if !size_fits || is_wrong_sign {
        return FitsIn::No {
            need: smallest_fit_size.map(|sfs| {
                NoirType::Integer(if is_wrong_sign { Signedness::Signed } else { *hole_sign }, sfs)
            }),
        };
    }

    return FitsIn::Yes;
}

pub fn type_infer(state: State, expr: SpannedExpr) -> Result<SpannedTypedExpr, TypeInferenceError> {
    let sote: SpannedOptionallyTypedExpr = try_contextual_cata(
        expr,
        vec![],
        &|mut quantifier_bound_variables, e| {
            if let ExprF::Quantified { name, .. } = e.expr.as_ref() {
                quantifier_bound_variables.push(name.clone())
            }
            quantifier_bound_variables
        },
        &|location, quantifier_bound_variables, exprf| {
            let (exprf, exprf_type) = match &exprf {
                ExprF::Literal { value } => match value {
                    Literal::Bool(_) => (exprf, Some(NoirType::Bool)),
                    Literal::Int(_) => {
                        // NOTE: `None` signifies that this has to be inferred up the chain, will gain
                        //       a concrete type when it gets matched (in an arithmetic or predicate)
                        //       operation with a variable with a real (integer) type
                        (exprf, None)
                    }
                },
                ExprF::Variable { name } => {
                    let (variable_ident, variable_type): (&str, Option<NoirType>) =
                        quantifier_bound_variables
                            .iter()
                            .find_map(|bound_variable| {
                                (bound_variable == name).then(|| (name.as_str(), None))
                            })
                            .or_else(|| {
                                state.function.parameters.iter().find_map(|k| {
                                    (k.2 == *name).then(|| (name.as_str(), Some(k.3.clone())))
                                })
                            })
                            .or_else(|| {
                                (name == "result").then(|| {
                                    (FUNC_RETURN_VAR_NAME, Some(state.function.return_type.clone()))
                                })
                            })
                            .ok_or(TypeInferenceError::TypeError {
                                got: None,
                                wanted: None,
                                message: Some(format!("Undefined variable {}", name)),
                            })?;

                    (ExprF::Variable { name: variable_ident.to_string() }, variable_type)
                }
                ExprF::FnCall { name, args } => {
                    let return_type = state
                        .functions
                        .iter()
                        .find_map(|(_, func)| (func.name == *name).then_some(&func.return_type))
                        .ok_or(TypeInferenceError::MonomorphizationRequest(
                            MonomorphizationRequest {
                                function_identifier: name.clone(),
                                param_types: args
                                    .iter()
                                    .map(|arg: &SpannedOptionallyTypedExpr| arg.ann.1.clone())
                                    .collect(),
                            },
                        ))?;

                    (exprf, Some(return_type.clone()))
                }
                ExprF::Quantified { .. } => (exprf, Some(NoirType::Bool)),
                ExprF::Parenthesised { expr } => (exprf.clone(), expr.ann.1.clone()),
                ExprF::UnaryOp { op: _, expr } => (exprf.clone(), expr.ann.1.clone()),
                ExprF::BinaryOp { op, expr_left, expr_right } => {
                    match op {
                        BinaryOp::Mul
                        | BinaryOp::Div
                        | BinaryOp::Mod
                        | BinaryOp::Add
                        | BinaryOp::Sub
                        | BinaryOp::ShiftLeft
                        | BinaryOp::ShiftRight
                        | BinaryOp::Eq
                        | BinaryOp::Neq
                        | BinaryOp::Lt
                        | BinaryOp::Le
                        | BinaryOp::Gt
                        | BinaryOp::Ge
                        | BinaryOp::And
                        | BinaryOp::Or
                        | BinaryOp::Xor => {
                            let is_arith = op.is_arithmetic();

                            let propagate_concrete_type =
                                |e: SpannedOptionallyTypedExpr, t: NoirType| {
                                    let NoirType::Integer(hole_sign, hole_size) = &t else {
                                        todo!("Can only propagate integer types");
                                    };
                                    try_cata(e, &|(location, _type), expr| {
                                        debug_assert!(_type == None);
                                        match expr {
                                            ExprF::Literal { value: Literal::Int(ref bi) } => {
                                                let fits = bi_can_fit_in(bi, hole_size, hole_sign);
                                                match fits {
                                                    FitsIn::Yes => {}
                                                    FitsIn::No { need } => {
                                                        return Err(
                                                            TypeInferenceError::TypeError {
                                                                got: need.clone(),
                                                                wanted: Some(t.clone()),
                                                                message: Some(format!(
                                                                    "Integer literal {} cannot fit in {}, needs at least {:?} or larger",
                                                                    bi, t, need,
                                                                )),
                                                            },
                                                        );
                                                    }
                                                }
                                            }
                                            _ => {}
                                        }

                                        Ok(SpannedOptionallyTypedExpr {
                                            expr: Box::new(expr),
                                            ann: (location, Some(t.clone())),
                                        })
                                    })
                                };

                            match (&expr_left.ann.1, &expr_right.ann.1) {
                                (None, None) => (exprf, None),
                                (None, Some(t2)) => {
                                    let expr_left_inner =
                                        propagate_concrete_type(expr_left.clone(), t2.clone())?;

                                    (
                                        ExprF::BinaryOp {
                                            op: op.clone(),
                                            expr_left: expr_left_inner,
                                            expr_right: expr_right.clone(),
                                        },
                                        Some(if is_arith { t2.clone() } else { NoirType::Bool }),
                                    )
                                }
                                (Some(t1), None) => {
                                    let expr_right_inner =
                                        propagate_concrete_type(expr_right.clone(), t1.clone())?;

                                    (
                                        ExprF::BinaryOp {
                                            op: op.clone(),
                                            expr_left: expr_left.clone(),
                                            expr_right: expr_right_inner,
                                        },
                                        Some(if is_arith { t1.clone() } else { NoirType::Bool }),
                                    )
                                }
                                (Some(t1), Some(t2)) => {
                                    if t1 != t2 {
                                        return Err(TypeInferenceError::TypeError {
                                            got: Some(t2.clone()),
                                            wanted: Some(t1.clone()),
                                            message: Some(format!(
                                                "Different types of arguments to {} operation",
                                                if is_arith { "arithmetic" } else { "predicate" }
                                            )),
                                        });
                                    }

                                    (
                                        ExprF::BinaryOp {
                                            op: op.clone(),
                                            expr_left: expr_left.clone(),
                                            expr_right: expr_right.clone(),
                                        },
                                        Some(if is_arith { t1.clone() } else { NoirType::Bool }),
                                    )
                                }
                            }
                        }

                        // pure Boolean
                        BinaryOp::Implies => {
                            if expr_left.ann.1 != Some(NoirType::Bool) {
                                return Err(TypeInferenceError::TypeError {
                                    got: expr_left.ann.1.clone(),
                                    wanted: Some(NoirType::Bool),
                                    message: Some(
                                        "Boolean operations work on boolean arguments".to_string(),
                                    ),
                                });
                            }
                            if expr_right.ann.1 != Some(NoirType::Bool) {
                                return Err(TypeInferenceError::TypeError {
                                    got: expr_right.ann.1.clone(),
                                    wanted: Some(NoirType::Bool),
                                    message: Some(
                                        "Boolean operations work on boolean arguments".to_string(),
                                    ),
                                });
                            }

                            (exprf, Some(NoirType::Bool))
                        }
                    }
                }
            };

            Ok(SpannedOptionallyTypedExpr { ann: (location, exprf_type), expr: Box::new(exprf) })
        },
    )?;

    let fully_typed_expr: SpannedTypedExpr =
        try_cata(sote, &|(location, otype), exprf| match otype {
            Some(t) => Ok(SpannedTypedExpr { ann: (location, t), expr: Box::new(exprf) }),
            None => Err(()),
        })
        .expect("Typing should have either succeeded or have resulted in an expected error");

    Ok(fully_typed_expr)
}

#[cfg(test)]
mod tests {
    use noirc_errors::{Location, Span};
    use noirc_frontend::ast::IntegerBitSize;
    use noirc_frontend::shared::Signedness;
    use num_traits::Num;
    use std::convert::identity;

    use super::*;

    use crate::{Attribute, ast::Literal, parse::tests::*, parse_attribute};

    #[test]
    fn test_integer_types() {
        {
            let bi127 = BigInt::from_str_radix("127", 10).unwrap();
            let bi128 = BigInt::from_str_radix("128", 10).unwrap();
            let bi129 = BigInt::from_str_radix("129", 10).unwrap();
            let bin127 = BigInt::from_str_radix("-127", 10).unwrap();
            let bin128 = BigInt::from_str_radix("-128", 10).unwrap();
            let bin129 = BigInt::from_str_radix("-129", 10).unwrap();

            let hole_size = IntegerBitSize::Eight;
            let hole_sign = Signedness::Signed;

            let bi127_fit = bi_can_fit_in(&bi127, &hole_size, &hole_sign);
            let bi128_fit = bi_can_fit_in(&bi128, &hole_size, &hole_sign);
            let bi129_fit = bi_can_fit_in(&bi129, &hole_size, &hole_sign);
            let bin127_fit = bi_can_fit_in(&bin127, &hole_size, &hole_sign);
            let bin128_fit = bi_can_fit_in(&bin128, &hole_size, &hole_sign);
            let bin129_fit = bi_can_fit_in(&bin129, &hole_size, &hole_sign);

            assert_eq!(bi127_fit, FitsIn::Yes);
            assert_eq!(
                bi128_fit,
                FitsIn::No { need: Some(NoirType::Integer(hole_sign, IntegerBitSize::Sixteen)) }
            );
            assert_eq!(
                bi129_fit,
                FitsIn::No { need: Some(NoirType::Integer(hole_sign, IntegerBitSize::Sixteen)) }
            );

            assert_eq!(bin127_fit, FitsIn::Yes);
            assert_eq!(bin128_fit, FitsIn::Yes);
            assert_eq!(
                bin129_fit,
                FitsIn::No { need: Some(NoirType::Integer(hole_sign, IntegerBitSize::Sixteen)) }
            );
        }

        {
            let bi255 = BigInt::from_str_radix("255", 10).unwrap();
            let bi256 = BigInt::from_str_radix("256", 10).unwrap();
            let bi257 = BigInt::from_str_radix("257", 10).unwrap();
            let bin1 = BigInt::from_str_radix("-1", 10).unwrap();

            let hole_size = IntegerBitSize::Eight;
            let hole_sign = Signedness::Unsigned;

            let bi255_fit = bi_can_fit_in(&bi255, &hole_size, &hole_sign);
            let bi256_fit = bi_can_fit_in(&bi256, &hole_size, &hole_sign);
            let bi257_fit = bi_can_fit_in(&bi257, &hole_size, &hole_sign);
            let bin1_fit = bi_can_fit_in(&bin1, &hole_size, &hole_sign);

            assert_eq!(bi255_fit, FitsIn::Yes);
            assert_eq!(
                bi256_fit,
                FitsIn::No { need: Some(NoirType::Integer(hole_sign, IntegerBitSize::Sixteen)) }
            );
            assert_eq!(
                bi257_fit,
                FitsIn::No { need: Some(NoirType::Integer(hole_sign, IntegerBitSize::Sixteen)) }
            );

            assert_eq!(
                bin1_fit,
                FitsIn::No {
                    need: Some(NoirType::Integer(Signedness::Signed, IntegerBitSize::One))
                }
            );
        }
    }

    #[test]
    fn test_whole_attribute() {
        let attribute = "ensures(result >= a + (16 / 2 % (7 * 4)))";
        let state = empty_state();
        let attribute = parse_attribute(
            attribute,
            Location { span: Span::inclusive(0, attribute.len() as u32), file: Default::default() },
            state.function,
            state.global_constants,
            state.functions,
        )
        .unwrap();
        let Attribute::Ensures(spanned_expr) = attribute else { panic!() };
        let spanned_typed_expr = type_infer(state, spanned_expr).unwrap();
        assert!(
            cata(spanned_typed_expr, &|(_, expr_type), expr| {
                match expr {
                    ExprF::Literal { value: Literal::Int(_) } => {
                        expr_type
                            == NoirType::Integer(Signedness::Signed, IntegerBitSize::ThirtyTwo)
                    }
                    ExprF::FnCall { args, .. } => args.into_iter().all(identity),
                    ExprF::Quantified { expr, .. } => expr,
                    ExprF::Parenthesised { expr } => expr,
                    ExprF::UnaryOp { expr, .. } => expr,
                    ExprF::BinaryOp { expr_left, expr_right, .. } => expr_left && expr_right,

                    // Non-recursive variants don't carry information
                    ExprF::Literal { value: Literal::Bool(_) } | ExprF::Variable { .. } => true,
                }
            }),
            "All integer literals have the correct inferred type"
        );
    }

    #[test]
    fn test_quantifiers() {
        let attribute = "ensures(exists(|i| result >= a + (16 / i % (7 * 4))))";
        let state = empty_state();
        let attribute = parse_attribute(
            attribute,
            Location { span: Span::inclusive(0, attribute.len() as u32), file: Default::default() },
            state.function,
            state.global_constants,
            state.functions,
        )
        .unwrap();
        let Attribute::Ensures(spanned_expr) = attribute else { panic!() };
        let spanned_typed_expr = type_infer(state, spanned_expr).unwrap();
        assert!(
            cata(spanned_typed_expr.clone(), &|(_, expr_type), expr| {
                match expr {
                    ExprF::Literal { value: Literal::Int(_) } => {
                        expr_type
                            == NoirType::Integer(Signedness::Signed, IntegerBitSize::ThirtyTwo)
                    }

                    ExprF::FnCall { args, .. } => args.into_iter().all(identity),
                    ExprF::Quantified { expr, .. } => expr,
                    ExprF::Parenthesised { expr } => expr,
                    ExprF::UnaryOp { expr, .. } => expr,
                    ExprF::BinaryOp { expr_left, expr_right, .. } => expr_left && expr_right,

                    // Non-recursive variants don't carry information
                    ExprF::Literal { value: Literal::Bool(_) } | ExprF::Variable { .. } => true,
                }
            }),
            "All integer literals have the correct inferred type"
        );
        assert!(
            cata(spanned_typed_expr, &|(_, expr_type), expr| {
                match expr {
                    ExprF::Variable { name } => {
                        if name == "i" {
                            expr_type
                                == NoirType::Integer(Signedness::Signed, IntegerBitSize::ThirtyTwo)
                        } else {
                            true
                        }
                    }

                    ExprF::FnCall { args, .. } => args.into_iter().all(identity),
                    ExprF::Quantified { expr, .. } => expr,
                    ExprF::Parenthesised { expr } => expr,
                    ExprF::UnaryOp { expr, .. } => expr,
                    ExprF::BinaryOp { expr_left, expr_right, .. } => expr_left && expr_right,

                    // Non-recursive variants don't carry information
                    ExprF::Literal { .. } => true,
                }
            }),
            "All bound variables have the correct inferred type"
        );
    }

    #[test]
    fn test_monomorphization_request() {
        let attribute = "ensures(f(result))";
        let state = empty_state();
        let attribute = parse_attribute(
            attribute,
            Location { span: Span::inclusive(0, attribute.len() as u32), file: Default::default() },
            state.function,
            state.global_constants,
            state.functions,
        )
        .unwrap();
        let Attribute::Ensures(spanned_expr) = attribute else { panic!() };
        let type_inference_error = type_infer(state, spanned_expr).unwrap_err();
        let TypeInferenceError::MonomorphizationRequest(MonomorphizationRequest {
            function_identifier,
            param_types,
        }) = type_inference_error
        else {
            panic!()
        };
        assert_eq!(function_identifier, "f");
        assert_eq!(
            param_types,
            vec![Some(NoirType::Integer(Signedness::Signed, IntegerBitSize::ThirtyTwo))]
        );
    }
}
