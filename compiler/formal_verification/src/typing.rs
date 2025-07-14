use crate::{
    MonomorphizationRequest, State,
    ast::{
        BinaryOp, ExprF, Literal, SpannedExpr, SpannedOptionallyTypedExpr, SpannedTypedExpr,
        UnaryOp, cata, try_cata,
    },
};
use noirc_frontend::monomorphization::{FUNC_RETURN_VAR_NAME, ast::Type as NoirType};

#[derive(Debug, Clone)]
pub enum TypeInferenceError {
    MonomorphizationRequest(MonomorphizationRequest),
    TypeError { got: Option<NoirType>, wanted: Option<NoirType>, message: Option<String> },
}

pub fn type_infer(state: State, expr: SpannedExpr) -> Result<SpannedTypedExpr, TypeInferenceError> {
    let sote: SpannedOptionallyTypedExpr = try_cata(expr, &|location, exprf| {
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
                let (variable_ident, variable_type): (&str, &NoirType) = state
                    .function
                    .parameters
                    .iter()
                    .find_map(|k| (k.2 == *name).then(|| (name.as_str(), &k.3)))
                    .or_else(|| {
                        (name == "result")
                            .then(|| (FUNC_RETURN_VAR_NAME, &state.function.return_type))
                    })
                    .ok_or(TypeInferenceError::TypeError {
                        got: None,
                        wanted: None,
                        message: Some(format!("Undefined variable {}", name)),
                    })?;

                (ExprF::Variable { name: variable_ident.to_string() }, Some(variable_type.clone()))
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
            ExprF::Parenthesised { expr } => (exprf.clone(), expr.ann.1.clone()),
            ExprF::UnaryOp { op, expr } => {
                let expr_type = match op {
                    UnaryOp::Not => {
                        // TODO: can work with non-booleans
                        if expr.ann.1 != Some(NoirType::Bool) {
                            return Err(TypeInferenceError::TypeError {
                                got: expr.ann.1.clone(),
                                wanted: Some(NoirType::Bool),
                                message: Some("Non-boolean passed to logical not".to_string()),
                            });
                        }

                        NoirType::Bool
                    }
                };

                (exprf, Some(expr_type))
            }
            ExprF::BinaryOp { op, expr_left, expr_right } => match op {
                BinaryOp::ArithmeticOp(_) | BinaryOp::PredicateOp(_) => {
                    let is_arith = match op {
                        BinaryOp::ArithmeticOp(_) => true,
                        BinaryOp::PredicateOp(_) => false,
                        _ => unreachable!(),
                    };
                    match (&expr_left.ann.1, &expr_right.ann.1) {
                        (None, None) => (exprf, None),
                        (None, Some(t2)) => {
                            let expr_left_inner =
                                cata(expr_left.clone(), &|(location, _type), expr| {
                                    debug_assert!(_type == None);
                                    SpannedOptionallyTypedExpr {
                                        expr: Box::new(expr),
                                        ann: (location, Some(t2.clone())),
                                    }
                                });

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
                                cata(expr_right.clone(), &|(location, _type), expr| {
                                    debug_assert!(_type == None);
                                    SpannedOptionallyTypedExpr {
                                        expr: Box::new(expr),
                                        ann: (location, Some(t1.clone())),
                                    }
                                });

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
                BinaryOp::BooleanOp(_) => {
                    // TODO: can work with non-booleans (except implication)
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
            },
        };

        Ok(SpannedOptionallyTypedExpr { ann: (location, exprf_type), expr: Box::new(exprf) })
    })?;

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
    use noirc_frontend::ast::IntegerBitSize;
    use noirc_frontend::shared::Signedness;
    use std::convert::identity;

    use super::*;

    use crate::{Attribute, ast::Literal, parse::tests::*, parse_attribute};

    #[test]
    fn test_whole_attribute() {
        let attribute = "ensures(result >= a + (16 / 2 % (7 * 4)))";
        let state = empty_state(attribute.len() as u32);
        let attribute = parse_attribute(
            attribute,
            state.location,
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
    fn test_monomorphization_request() {
        let attribute = "ensures(f(result))";
        let state = empty_state(attribute.len() as u32);
        let attribute = parse_attribute(
            attribute,
            state.location,
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
