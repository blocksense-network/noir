use std::collections::HashMap;

use noirc_frontend::{
    ast::Ident,
    hir::{resolution::errors::ResolverError, type_check::TypeCheckError},
};

use crate::ast::{ExprF, SpannedExpr, try_cata_mut};

pub fn convert_struct_access_to_tuple_access(
    expr: SpannedExpr,
    parameters_hir_types: &HashMap<String, noirc_frontend::Type>,
) -> Result<SpannedExpr, ResolverOrTypeError> {
    let mut last_important_type: Option<noirc_frontend::Type> = None;

    let res: SpannedExpr =
        try_cata_mut(expr.clone(), &mut |loc, exprf| -> Result<_, ResolverOrTypeError> {
            match exprf {
                ExprF::Index { .. } => {
                    if let Some(last_type) = &last_important_type {
                        if let noirc_frontend::Type::Array(_size, inner_type) = last_type {
                            last_important_type = Some(*inner_type.clone());
                            Ok(SpannedExpr { ann: loc, expr: Box::new(exprf) })
                        } else {
                            // Error, tried to index a non array type
                            Err(ResolverOrTypeError::TypeError(TypeCheckError::TypeMismatch {
                                expr_typ: last_type.to_string(),
                                expected_typ: String::from("array"),
                                expr_location: loc,
                            }))
                        }
                    } else {
                        Ok(SpannedExpr { ann: loc, expr: Box::new(exprf) })
                    }
                }
                ExprF::TupleAccess { ref index, .. } => {
                    if let Some(last_type) = &last_important_type {
                        if let noirc_frontend::Type::Tuple(inner_types) = last_type {
                            last_important_type = Some(inner_types[*index as usize].clone());
                            Ok(SpannedExpr { ann: loc, expr: Box::new(exprf) })
                        } else {
                            // Error, tried to index a non tuple
                            Err(ResolverOrTypeError::TypeError(TypeCheckError::TypeMismatch {
                                expr_typ: last_type.to_string(),
                                expected_typ: String::from("tuple"),
                                expr_location: loc,
                            }))
                        }
                    } else {
                        Ok(SpannedExpr { ann: loc, expr: Box::new(exprf) })
                    }
                }
                ExprF::StructureAccess { expr, field } => {
                    if let Some(last_type) = last_important_type.take() {
                        if let noirc_frontend::Type::DataType(struct_type, _) = last_type {
                            if let Some((typ, _, index)) =
                                struct_type.borrow().get_field(&field, &[])
                            {
                                last_important_type = Some(typ);
                                Ok(SpannedExpr {
                                    ann: loc,
                                    expr: Box::new(ExprF::TupleAccess {
                                        expr: expr,
                                        index: index as u32,
                                    }),
                                })
                            } else {
                                // Error tried to access a non existing field of a structure
                                Err(ResolverOrTypeError::ResolverError(
                                    ResolverError::NoSuchField {
                                        field: Ident::new(field, loc),
                                        struct_definition: struct_type.borrow().name.clone(),
                                    },
                                ))
                            }
                        } else {
                            // Error, tried to access a field of a non struct type
                            Err(ResolverOrTypeError::TypeError(TypeCheckError::TypeMismatch {
                                expr_typ: last_type.to_string(),
                                expected_typ: String::from("structure"),
                                expr_location: loc,
                            }))
                        }
                    } else {
                        // Error, can't convert structure access to tuple access because
                        // of missing type information for inner expression
                        // Should be unreachable
                        unreachable!()
                    }
                }
                ExprF::Variable(ref var) => {
                    if let Some(var_typ) = parameters_hir_types.get(&var.name) {
                        last_important_type = Some(var_typ.clone());
                    } else {
                        // NOTE: We are processing a quantifier index, therefore we don't have
                        // to keep track of its type because it's an integer
                        last_important_type = None;
                    }
                    Ok(SpannedExpr { ann: loc, expr: Box::new(exprf) })
                }
                ExprF::Parenthesised { .. }
                | ExprF::UnaryOp { op: crate::ast::UnaryOp::Dereference, .. } => {
                    Ok(SpannedExpr { ann: loc, expr: Box::new(exprf) })
                }
                _ => {
                    last_important_type = None;
                    Ok(SpannedExpr { ann: loc, expr: Box::new(exprf) })
                }
            }
        })?;

    Ok(res)
}

pub enum ResolverOrTypeError {
    ResolverError(ResolverError),
    TypeError(TypeCheckError),
}
