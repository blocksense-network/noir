use noirc_errors::Location;
use noirc_frontend::{
    ast::Ident,
    hir::{
        comptime::Value,
        resolution::{errors::ResolverError, import::PathResolutionError},
    },
    node_interner::GlobalValue,
};
use num_bigint::{BigInt, BigUint, Sign};
use std::collections::HashMap;

use crate::ast::{AnnExpr, ExprF, Literal, SpannedExpr, try_cata};

/// Replaces all global identifiers with their resolved constant value.
pub fn inline_global_consts(
    expr: SpannedExpr,
    pathed_globals_with_values: &HashMap<String, GlobalValue>,
    function_parameters: &[&str],
) -> Result<SpannedExpr, ResolverError> {
    try_cata(expr, &|loc, exprf| -> Result<_, ResolverError> {
        match exprf {
            ExprF::Variable(variable) => {
                // --- Guard 1: Handle function parameters first ---
                // Parameters have the highest precedence and are never inlined.
                let is_parameter = variable.path.is_empty()
                    && function_parameters.contains(&variable.name.as_str());

                if is_parameter {
                    // It's a parameter. Leave it as is and we're done with this variable.
                    return Ok(SpannedExpr { ann: loc, expr: Box::new(ExprF::Variable(variable)) });
                }

                // --- Guard 2: Try to resolve as a global constant ---
                // At this point, we know it's not a shadowed parameter.
                let full_path = join_path_segments(variable.path.clone(), variable.name.clone());
                if let Some(global_val) = pathed_globals_with_values.get(&full_path) {
                    // It's a valid global. Inline it.
                    return match global_val {
                        GlobalValue::Unresolved | GlobalValue::Resolving => {
                            unreachable!("All global constants must have been resolved by now")
                        }
                        GlobalValue::Resolved(value) => {
                            let global_const_as_exprf = resolved_value_to_exprf(value, loc)?;
                            Ok(SpannedExpr { ann: loc, expr: Box::new(global_const_as_exprf) })
                        }
                    };
                }

                // --- Guard 3: Check for invalid, unresolved paths ---
                // It's not a parameter and not a known global. If it has a path, it's an error.
                if !variable.path.is_empty() {
                    return Err(ResolverError::PathResolutionError(
                        PathResolutionError::Unresolved(Ident::new(full_path, loc)),
                    ));
                }

                // --- Default Case ---
                // If none of the guards above returned, the variable must be a local identifier
                // that is not a parameter (e.g., `result`, quantifier, or undeclared variable).
                // We leave it as is for a later compiler stage to handle.
                Ok(SpannedExpr { ann: loc, expr: Box::new(ExprF::Variable(variable)) })
            }
            // For any other expression type, just reconstruct it.
            _ => Ok(SpannedExpr { ann: loc, expr: Box::new(exprf) }),
        }
    })
}

fn resolved_value_to_exprf(
    value: &Value,
    location: Location,
) -> Result<ExprF<AnnExpr<Location>>, ResolverError> {
    Ok(match value {
        Value::Unit => ExprF::Tuple { exprs: Vec::new() },
        Value::Bool(bool_val) => ExprF::Literal { value: Literal::Bool(*bool_val) },
        Value::Field(signed_field) => {
            let field_as_big_uint: BigUint = signed_field.to_field_element().into_repr().into();
            ExprF::Literal {
                value: Literal::Int(BigInt::from_biguint(Sign::Plus, field_as_big_uint)),
            }
        }
        Value::I8(integer) => ExprF::Literal { value: Literal::Int(BigInt::from(*integer)) },
        Value::I16(integer) => ExprF::Literal { value: Literal::Int(BigInt::from(*integer)) },
        Value::I32(integer) => ExprF::Literal { value: Literal::Int(BigInt::from(*integer)) },
        Value::I64(integer) => ExprF::Literal { value: Literal::Int(BigInt::from(*integer)) },
        Value::U1(integer) => ExprF::Literal { value: Literal::Int(BigInt::from(*integer)) },
        Value::U8(integer) => ExprF::Literal { value: Literal::Int(BigInt::from(*integer)) },
        Value::U16(integer) => ExprF::Literal { value: Literal::Int(BigInt::from(*integer)) },
        Value::U32(integer) => ExprF::Literal { value: Literal::Int(BigInt::from(*integer)) },
        Value::U64(integer) => ExprF::Literal { value: Literal::Int(BigInt::from(*integer)) },
        Value::U128(integer) => ExprF::Literal { value: Literal::Int(BigInt::from(*integer)) },
        Value::Tuple(values) => {
            let exprs = values
                .iter()
                .map(|val| {
                    let exprf = resolved_value_to_exprf(val, location)?;
                    Ok(SpannedExpr { ann: location, expr: Box::new(exprf) })
                })
                .collect::<Result<_, ResolverError>>()?;
            ExprF::Tuple { exprs }
        }
        Value::Array(values, _) => {
            let exprs = values
                .iter()
                .map(|val| {
                    let exprf = resolved_value_to_exprf(val, location)?;
                    Ok(SpannedExpr { ann: location, expr: Box::new(exprf) })
                })
                .collect::<Result<_, ResolverError>>()?;
            ExprF::Array { exprs }
        }
        Value::String(..)
        | Value::FormatString(..)
        | Value::CtString(..)
        // We currently don't support strings in formal verification
        | Value::Function(..)
        | Value::Closure(..)
        | Value::Struct(..)
        | Value::Enum(..)
        | Value::Pointer(..)
        | Value::Slice(..)
        | Value::Quoted(..)
        | Value::TypeDefinition(..)
        | Value::TraitConstraint(..)
        | Value::TraitDefinition(..)
        | Value::TraitImpl(..)
        | Value::FunctionDefinition(..)
        | Value::ModuleDefinition(..)
        | Value::Type(..)
        | Value::Zeroed(..)
        | Value::Expr(..)
        | Value::TypedExpr(..)
        | Value::UnresolvedType(..) => {
            // We believe that all global const values are resolved
            return Err(ResolverError::UnevaluatedGlobalType { location });
        }
    })
}

/// Joins path segments and an identifier into a single "::" delimited string.
///
/// # Arguments
/// * `path_parts` - A vector of strings for the path's base (e.g., ["super", "foo"]).
/// * `identifier` - The final identifier to append (e.g., "bar").
///
/// # Returns
/// A single string like "super::foo::bar".
fn join_path_segments(mut path_parts: Vec<String>, identifier: String) -> String {
    path_parts.push(identifier);
    path_parts.join("::")
}
