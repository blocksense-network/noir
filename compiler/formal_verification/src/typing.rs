use crate::{
    MonomorphizationRequest, State,
    ast::{
        BinaryOp, ExprF, Literal, SpannedExpr, SpannedOptionallyTypedExpr, SpannedTypedExpr,
        UnaryOp, Variable, strip_ann, try_cata, try_contextual_cata,
    },
};
use noirc_errors::Location;
use noirc_frontend::{
    ast::IntegerBitSize,
    hir::{resolution::errors::ResolverError, type_check::TypeCheckError},
    monomorphization::{FUNC_RETURN_VAR_NAME, ast::Type as NoirType},
    shared::Signedness,
};
use num_bigint::{BigInt, BigUint};
use num_traits::{One, Zero};

#[derive(Debug, Clone)]
pub enum TypeInferenceError {
    MonomorphizationRequest(MonomorphizationRequest),
    // NOTE: We are converting IntegerLiteralDoesNotFit to TypeCheckError later
    // We cannot do it during construction because we need a function
    // located in a module which depends on us.
    IntegerLiteralDoesNotFit {
        literal: BigInt,
        literal_type: NoirType,
        fit_into: Option<NoirType>,
        location: Location,
        message: String,
    },
    NoirTypeError(TypeCheckError),
}

#[derive(Debug, PartialEq, Eq)]
pub enum FitsIn {
    Yes,
    No {
        /// `None` means there is no possible `IntegerBitSize` that could contain us
        need: Option<NoirType>,
    },
}
pub fn bi_can_fit_in(bi: &BigInt, hole_size: &IntegerBitSize, hole_sign: &Signedness) -> FitsIn {
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

// WARN: will (possibly) incorrectly propagate types in tuple literals
//       `(1000, 5, 1000).1 == (5 as u8)`
pub fn propagate_concrete_type(
    e: SpannedOptionallyTypedExpr,
    t: NoirType,
) -> Result<SpannedOptionallyTypedExpr, TypeInferenceError> {
    let limits = match &t {
        NoirType::Field => None,
        NoirType::Integer(hole_sign, hole_size) => Some((hole_sign, hole_size)),
        _ => todo!(
            "Can only propagate integer types, trying to ram {:#?} into {:#?}",
            t,
            strip_ann(e)
        ),
    };

    try_cata(e, &|(location, _type), expr| {
        match expr {
            ExprF::Literal { value: Literal::Int(ref bi) } => {
                debug_assert!(
                    _type == None,
                    "Trying to smash type {:?} into {:?} which already has type {:?}",
                    t,
                    bi,
                    _type
                );
                // NOTE: only check limits for integer types
                //       (assume that `NoirType::Field` can hold anything)
                if let Some((hole_sign, hole_size)) = limits {
                    let fits = bi_can_fit_in(bi, hole_size, hole_sign);
                    match fits {
                        FitsIn::Yes => {}
                        FitsIn::No { need } => {
                            return Err(TypeInferenceError::IntegerLiteralDoesNotFit {
                                literal: bi.clone(),
                                literal_type: t.clone(),
                                fit_into: need.clone(),
                                message: format!(
                                    "Integer literal {} cannot fit in {}, needs at least {:?} or larger",
                                    bi, t, need,
                                ),
                                location,
                            });
                        }
                    }
                }
            }
            _ => {}
        }

        Ok(SpannedOptionallyTypedExpr { expr: Box::new(expr), ann: (location, Some(t.clone())) })
    })
}

pub fn type_infer(
    state: &State,
    expr: SpannedExpr,
) -> Result<SpannedTypedExpr, TypeInferenceError> {
    // NOTE: predicate, always bool,
    //       assume subterms are `u32` (like `Noir` does)
    let default_literal_type = NoirType::Integer(Signedness::Unsigned, IntegerBitSize::ThirtyTwo);

    let is_numeric = |t: &NoirType| matches!(t, NoirType::Integer(_, _) | NoirType::Field);

    let sote: SpannedOptionallyTypedExpr = try_contextual_cata(
        expr,
        vec![],
        &|mut quantifier_bound_variables, e| {
            if let ExprF::Quantified { variables, .. } = e.expr.as_ref() {
                quantifier_bound_variables.extend(variables.iter().map(
                    |Variable { path, name, .. }| {
                        // NOTE: quantified variables should have no path
                        debug_assert_eq!(path.len(), 0);

                        name.clone()
                    },
                ));
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
                ExprF::Variable(Variable { path, name, id }) => {
                    // TODO: we do not support type inferrence of variables with paths
                    debug_assert_eq!(path.len(), 0);
                    // NOTE: parsing should not yield `id`s
                    debug_assert_eq!(*id, None);
                    let (variable_ident, variable_id, variable_type): (
                        &str,
                        Option<u32>,
                        Option<NoirType>,
                    ) = quantifier_bound_variables
                        .iter()
                        .find_map(|bound_variable| {
                            // TODO: `id` not `None` (when we have a way to generate new `id`s)
                            (bound_variable == name).then(|| (name.as_str(), None, None))
                        })
                        .or_else(|| {
                            state.function.parameters.iter().find_map(|(id, _, par_name, t, _)| {
                                (par_name == name)
                                    .then(|| (name.as_str(), Some(id.0), Some(t.clone())))
                            })
                        })
                        .or_else(|| {
                            (name == "result").then(|| {
                                (
                                    FUNC_RETURN_VAR_NAME,
                                    None,
                                    Some(state.function.return_type.clone()),
                                )
                            })
                        })
                        .ok_or(TypeInferenceError::NoirTypeError(TypeCheckError::ResolverError(
                            ResolverError::VariableNotDeclared { name: name.to_string(), location },
                        )))?;

                    (
                        ExprF::Variable(Variable {
                            path: path.clone(),
                            name: variable_ident.to_string(),
                            id: variable_id,
                        }),
                        variable_type,
                    )
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
                ExprF::Quantified { variables, .. } => {
                    variables
                        .iter()
                        .map(|Variable { path, .. }| {
                            if !path.is_empty() {
                                Err(TypeInferenceError::NoirTypeError(
                                    // TODO(totel): better error?
                                    TypeCheckError::ParameterCountMismatch {
                                        expected: 0,
                                        found: path.len(),
                                        location,
                                    },
                                ))
                            } else {
                                Ok(())
                            }
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                    (exprf, Some(NoirType::Bool))
                }
                ExprF::Parenthesised { expr } => (exprf.clone(), expr.ann.1.clone()),
                ExprF::UnaryOp { op, expr } => {
                    let t = match op {
                        UnaryOp::Dereference => {
                            match &expr.ann.1 {
                                Some(NoirType::Reference(t, _)) => Some(*t.clone()),
                                // TODO(totel): better error?
                                Some(_) | None => {
                                    return Err(TypeInferenceError::NoirTypeError(
                                        TypeCheckError::UnconstrainedReferenceToConstrained {
                                            location,
                                        },
                                    ));
                                }
                            }
                        }
                        UnaryOp::Not => expr.ann.1.clone(),
                    };

                    (exprf.clone(), t)
                }
                ExprF::BinaryOp { op, expr_left, expr_right } => {
                    match op {
                        BinaryOp::ShiftLeft | BinaryOp::ShiftRight => {
                            let mut expr_left = expr_left.clone();
                            let mut expr_right = expr_right.clone();

                            let shift_amount_type =
                                NoirType::Integer(Signedness::Unsigned, IntegerBitSize::Eight);

                            match &expr_right.ann.1 {
                                // Fine shift amount, only `u8` is allowed in `Noir`
                                Some(t) if *t == shift_amount_type => {}
                                // Integer literal, try type inferring to `u8`
                                None => {
                                    expr_right =
                                        propagate_concrete_type(expr_right, shift_amount_type)?;
                                }
                                // Not fine shift amount
                                Some(_) => {
                                    return Err(TypeInferenceError::NoirTypeError(
                                        TypeCheckError::InvalidShiftSize { location },
                                    ));
                                }
                            }

                            match &expr_left.ann.1 {
                                // Fine shiftee
                                Some(NoirType::Integer(_, _)) => {}
                                // Integer literal, try type inferring to u32
                                None => {
                                    expr_left = propagate_concrete_type(
                                        expr_left,
                                        default_literal_type.clone(),
                                    )?;
                                }
                                // Not fine shiftee
                                Some(expr_left_typ) => {
                                    return Err(TypeInferenceError::NoirTypeError(
                                        TypeCheckError::TypeMismatch {
                                            expr_typ: expr_left_typ.to_string(),
                                            expected_typ: String::from("Numeric type"),
                                            expr_location: location,
                                        },
                                    ));
                                }
                            }

                            let t = expr_left.ann.1.clone();

                            (ExprF::BinaryOp { op: op.clone(), expr_left, expr_right }, t)
                        }
                        BinaryOp::Mul
                        | BinaryOp::Div
                        | BinaryOp::Mod
                        | BinaryOp::Add
                        | BinaryOp::Sub
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

                            match (&expr_left.ann.1, &expr_right.ann.1) {
                                (None, None) => {
                                    if is_arith {
                                        (exprf, None)
                                    } else {
                                        let expr_left_inner = propagate_concrete_type(
                                            expr_left.clone(),
                                            default_literal_type.clone(),
                                        )?;
                                        let expr_right_inner = propagate_concrete_type(
                                            expr_right.clone(),
                                            default_literal_type.clone(),
                                        )?;

                                        (
                                            ExprF::BinaryOp {
                                                op: op.clone(),
                                                expr_left: expr_left_inner,
                                                expr_right: expr_right_inner,
                                            },
                                            Some(NoirType::Bool),
                                        )
                                    }
                                }
                                (None, Some(t2)) => {
                                    // NOTE: `1 & true`
                                    if is_arith && !is_numeric(t2) {
                                        return Err(TypeInferenceError::NoirTypeError(
                                            TypeCheckError::TypeMismatch {
                                                expected_typ: "Numeric type".to_string(),
                                                expr_typ: t2.to_string(),
                                                expr_location: location,
                                            },
                                        ));
                                    }

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
                                    // NOTE: `true & 1`
                                    if is_arith && !is_numeric(t1) {
                                        return Err(TypeInferenceError::NoirTypeError(
                                            TypeCheckError::TypeMismatch {
                                                expected_typ: "Numeric type".to_string(),
                                                expr_typ: t1.to_string(),
                                                expr_location: location,
                                            },
                                        ));
                                    }

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
                                        return Err(TypeInferenceError::NoirTypeError(
                                            TypeCheckError::TypeMismatch {
                                                expected_typ: t2.to_string(),
                                                expr_typ: t1.to_string(),
                                                expr_location: location,
                                            },
                                        ));
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
                                return Err(TypeInferenceError::NoirTypeError(
                                    TypeCheckError::TypeMismatch {
                                        expr_typ: expr_left
                                            .ann
                                            .1
                                            .as_ref()
                                            .map(|t| t.to_string())
                                            .unwrap_or(String::new()),
                                        expected_typ: NoirType::Bool.to_string(),
                                        expr_location: location,
                                    },
                                ));
                            }
                            if expr_right.ann.1 != Some(NoirType::Bool) {
                                return Err(TypeInferenceError::NoirTypeError(
                                    TypeCheckError::TypeMismatch {
                                        expr_typ: expr_right
                                            .ann
                                            .1
                                            .as_ref()
                                            .map(|t| t.to_string())
                                            .unwrap_or(String::new()),
                                        expected_typ: NoirType::Bool.to_string(),
                                        expr_location: location,
                                    },
                                ));
                            }

                            (exprf, Some(NoirType::Bool))
                        }
                    }
                }
                ExprF::Index { expr, index } => {
                    let mut index = index.clone();

                    let Some(NoirType::Array(_, type_inner)) = &expr.ann.1 else {
                        return Err(TypeInferenceError::NoirTypeError(
                            TypeCheckError::TypeMismatch {
                                expr_typ: expr
                                    .ann
                                    .1
                                    .as_ref()
                                    .map(|t| t.to_string())
                                    .unwrap_or(String::new()),
                                expected_typ: String::from("Array type"),
                                expr_location: location,
                            },
                        ));
                    };
                    match &index.ann.1 {
                        // Fine index
                        Some(NoirType::Integer(Signedness::Unsigned, _)) => {}
                        // Integer literal, try type inferring to `u32`
                        None => {
                            index = propagate_concrete_type(index, default_literal_type.clone())?;
                        }
                        // Not fine index
                        Some(_) => {
                            return Err(TypeInferenceError::NoirTypeError(
                                TypeCheckError::TypeMismatch {
                                    expr_typ: index
                                        .ann
                                        .1
                                        .as_ref()
                                        .map(|t| t.to_string())
                                        .unwrap_or(String::new()),
                                    expected_typ: String::from("Unsigned integer type"),
                                    expr_location: location,
                                },
                            ));
                        }
                    }

                    (ExprF::Index { expr: expr.clone(), index }, Some(*type_inner.clone()))
                }
                ExprF::TupleAccess { expr, index } => {
                    let Some(NoirType::Tuple(types)) = &expr.ann.1 else {
                        // TODO(totel): better error?
                        return Err(TypeInferenceError::NoirTypeError(
                            TypeCheckError::ResolverError(ResolverError::SelfReferentialType {
                                location,
                            }),
                        ));
                    };
                    let type_inner = types.get(*index as usize).ok_or(
                        TypeInferenceError::NoirTypeError(TypeCheckError::TupleIndexOutOfBounds {
                            index: *index as usize,
                            // NOTE: We are converting to Type::Tuple of empty vec because
                            // the inner types don't really matter for the error reporting
                            lhs_type: noirc_frontend::Type::Tuple(vec![]),
                            length: types.len(),
                            location,
                        }),
                    )?;

                    (exprf.clone(), Some(type_inner.clone()))
                }
                ExprF::Cast { expr, target } => {
                    let mut expr = expr.clone();

                    // Non-booleans cannot cast to bool
                    if matches!(target, NoirType::Bool)
                        && !matches!(expr.ann.1, Some(NoirType::Bool))
                    {
                        return Err(TypeInferenceError::NoirTypeError(
                            TypeCheckError::CannotCastNumericToBool {
                                // NOTE: We are using a mock type, because we can't convert to wanted type
                                typ: noirc_frontend::Type::FieldElement,
                                location,
                            },
                        ));
                    }

                    // Non-numerics cannot cast to numeric types
                    if is_numeric(target)
                        && let Some(ref t) = expr.ann.1
                        && !is_numeric(t)
                        && !matches!(t, NoirType::Bool)
                    {
                        return Err(TypeInferenceError::NoirTypeError(
                            TypeCheckError::TypeMismatch {
                                expected_typ: String::from("Numeric or a boolean type"),
                                expr_typ: t.to_string(),
                                expr_location: location,
                            },
                        ));
                    }

                    // Try to type infer integer literals as the target type
                    if matches!(expr.ann.1, None) {
                        expr = propagate_concrete_type(expr, target.clone())?;
                    }

                    (ExprF::Cast { expr, target: target.clone() }, Some(target.clone()))
                }
                ExprF::Tuple { exprs } => {
                    // TODO: support not-yet-typed expressions in the tuple literals,
                    //       later back-propagating the type inferrence through the projections
                    //       into the tuple
                    (
                        exprf.clone(),
                        Some(NoirType::Tuple(
                            exprs
                                .iter()
                                .map(|e| e.ann.1.clone())
                                .collect::<Option<Vec<_>>>()
                                .ok_or(TypeInferenceError::NoirTypeError(
                                    TypeCheckError::TypeAnnotationsNeededForFieldAccess {
                                        location,
                                    },
                                ))?,
                        )),
                    )
                }
            };

            Ok(SpannedOptionallyTypedExpr { ann: (location, exprf_type), expr: Box::new(exprf) })
        },
    )?;

    let fully_typed_expr: SpannedTypedExpr =
        try_cata(sote, &|(location, otype), exprf| match otype {
            Some(t) => Ok(SpannedTypedExpr { ann: (location, t), expr: Box::new(exprf) }),
            None => Err(format!("Expr {:?} still has no type", exprf)),
        })
        .expect("Typing should have either succeeded or have resulted in an expected error");

    // TODO: `assert!` that only the `FUNC_RETURN_VAR_NAME`

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

    use crate::{
        Attribute,
        ast::{Literal, Variable, cata},
        parse::{parse_attribute, tests::*},
    };

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
        let spanned_typed_expr = type_infer(&state, spanned_expr).unwrap();
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
                    ExprF::Index { expr, index } => expr && index,
                    ExprF::TupleAccess { expr, .. } => expr,
                    ExprF::Cast { expr, .. } => expr,
                    ExprF::Tuple { exprs } => exprs.into_iter().all(identity),

                    // Non-recursive variants don't carry information
                    ExprF::Literal { value: Literal::Bool(_) } | ExprF::Variable(_) => true,
                }
            }),
            "All integer literals have the correct inferred type"
        );
    }

    #[test]
    fn test_quantifiers() {
        let attribute = "ensures(exists(| i ,  j | result >= a + (16 / i % (7 * 4))))";
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
        let spanned_typed_expr = type_infer(&state, spanned_expr).unwrap();
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
                    ExprF::Index { expr, index } => expr && index,
                    ExprF::TupleAccess { expr, .. } => expr,
                    ExprF::Cast { expr, .. } => expr,
                    ExprF::Tuple { exprs } => exprs.into_iter().all(identity),

                    // Non-recursive variants don't carry information
                    ExprF::Literal { value: Literal::Bool(_) } | ExprF::Variable(_) => true,
                }
            }),
            "All integer literals have the correct inferred type"
        );
        assert!(
            cata(spanned_typed_expr, &|(_, expr_type), expr| {
                match expr {
                    ExprF::Variable(Variable { name, .. }) => {
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
                    ExprF::Index { expr, index } => expr && index,
                    ExprF::TupleAccess { expr, .. } => expr,
                    ExprF::Cast { expr, .. } => expr,
                    ExprF::Tuple { exprs } => exprs.into_iter().all(identity),

                    // Non-recursive variants don't carry information
                    ExprF::Literal { .. } => true,
                }
            }),
            "All bound variables have the correct inferred type"
        );
    }

    #[test]
    fn test_index() {
        let attribute = "ensures(xs[1 + 1] > 5)";
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
        let spanned_typed_expr = type_infer(&state, spanned_expr).unwrap();
        dbg!(&spanned_typed_expr);
        assert_eq!(spanned_typed_expr.ann.1, NoirType::Bool);
    }

    #[test]
    fn test_dereference_index() {
        let attribute = "ensures((*rxs)[1 + 1] > 5)";
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
        let spanned_typed_expr = type_infer(&state, spanned_expr).unwrap();
        dbg!(&spanned_typed_expr);
        assert_eq!(spanned_typed_expr.ann.1, NoirType::Bool);
    }

    #[test]
    fn test_operators_mixed_types() {
        let attribute = "ensures(1 + true)";
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
        let type_inference_error = type_infer(&state, spanned_expr).unwrap_err();
        let TypeInferenceError::NoirTypeError(TypeCheckError::TypeMismatch {
            expected_typ,
            expr_typ,
            expr_location,
        }) = type_inference_error
        else {
            panic!()
        };
        dbg!(&expr_typ, &expected_typ, &expr_location);

        // NOTE: untyped integer literal (same for quantifier variables) force the other argument
        //       to also be numeric
        assert_eq!(expr_typ, "bool");
        assert_eq!(expected_typ, "Numeric type");
    }

    #[test]
    fn test_bitshift() {
        let attribute = "ensures(1 << 256)";
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
        let type_inference_error = type_infer(&state, spanned_expr).unwrap_err();
        let TypeInferenceError::IntegerLiteralDoesNotFit {
            literal: _,
            literal_type,
            fit_into,
            location: _,
            message,
        } = type_inference_error
        else {
            panic!()
        };
        dbg!(&literal_type, &fit_into, &message);
        assert_eq!(literal_type, NoirType::Integer(Signedness::Unsigned, IntegerBitSize::Eight));
        // NOTE: minimal size that fits `256`
        assert_eq!(
            fit_into,
            Some(NoirType::Integer(Signedness::Unsigned, IntegerBitSize::Sixteen))
        );
        assert_eq!(message, "Integer literal 256 cannot fit in u8, needs at least Some(Integer(Unsigned, Sixteen)) or larger".to_string());
    }

    #[test]
    fn test_tuple_access() {
        let attribute = "ensures(user.0 ==> true)";
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
        let spanned_typed_expr = type_infer(&state, spanned_expr).unwrap();
        dbg!(&spanned_typed_expr);
        assert_eq!(spanned_typed_expr.ann.1, NoirType::Bool);
    }

    #[test]
    fn test_tuple_access_combos() {
        let attribute = "ensures(exists(|i| (0 <= i) & (i < 20) & xs[i] > 100))";
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
        let spanned_typed_expr = type_infer(&state, spanned_expr).unwrap();
        dbg!(&spanned_typed_expr);
        assert_eq!(spanned_typed_expr.ann.1, NoirType::Bool);
    }

    #[test]
    fn test_cast() {
        let annotation = "ensures((15 as i16 - 3 > 2) & ((result as Field - 6) as u64 == 1 + a as u64 >> 4 as u8))";
        let state = empty_state();
        let attribute = parse_attribute(
            annotation,
            Location {
                span: Span::inclusive(0, annotation.len() as u32),
                file: Default::default(),
            },
            state.function,
            state.global_constants,
            state.functions,
        )
        .unwrap();

        let Attribute::Ensures(spanned_expr) = attribute else { panic!() };
        let spanned_typed_expr = type_infer(&state, spanned_expr).unwrap();
        dbg!(&strip_ann(spanned_typed_expr));
        // assert_eq!(spanned_typed_expr.ann.1, NoirType::Bool);
    }

    #[test]
    fn test_double_cast() {
        let annotation = "ensures(a == (a as i16) as i32)";
        let state = empty_state();
        let attribute = parse_attribute(
            annotation,
            Location {
                span: Span::inclusive(0, annotation.len() as u32),
                file: Default::default(),
            },
            state.function,
            state.global_constants,
            state.functions,
        )
        .unwrap();

        let Attribute::Ensures(spanned_expr) = attribute else { panic!() };
        let spanned_typed_expr = type_infer(&state, spanned_expr).unwrap();
        dbg!(&strip_ann(spanned_typed_expr));
        // assert_eq!(spanned_typed_expr.ann.1, NoirType::Bool);
    }

    #[test]
    fn test_tuple() {
        let annotation = "ensures(((), kek, true).2)";
        let state = empty_state();
        let attribute = parse_attribute(
            annotation,
            Location {
                span: Span::inclusive(0, annotation.len() as u32),
                file: Default::default(),
            },
            state.function,
            state.global_constants,
            state.functions,
        )
        .unwrap();

        let Attribute::Ensures(spanned_expr) = attribute else { panic!() };
        let spanned_typed_expr = type_infer(&state, spanned_expr).unwrap();
        dbg!(&strip_ann(spanned_typed_expr));
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
        let type_inference_error = type_infer(&state, spanned_expr).unwrap_err();
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
