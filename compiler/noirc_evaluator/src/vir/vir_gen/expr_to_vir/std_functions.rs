use std::collections::BTreeMap;

use noirc_frontend::monomorphization::ast::{Call, Expression, GlobalId, Type};
use vir::ast::{Expr, ExprX, Mode, SpannedTyped};

use crate::vir::vir_gen::{
    build_span_no_id,
    expr_to_vir::{
        expr::{
            ast_definition_to_id, ast_expr_to_vir_expr, ast_ident_to_vir_var_ident,
            wrap_with_ghost_block,
        },
        types::{ast_type_to_vir_type, make_unit_vir_type},
    },
};

/// The function `assume()` from `fv_std_lib`
static ASSUME: &str = "assume";
/// The function `old()` from `fv_std_lib`
static OLD: &str = "old";

/// Handles function calls from the `fv_std` library and converts them to special VIR expressions.
pub fn handle_fv_std_call(
    call_expr: &Call,
    _mode: Mode, // Reserved for future use with additional `fv_std` functions
    globals: &BTreeMap<GlobalId, (String, Type, Expression)>,
) -> Option<Expr> {
    let ident = match &*call_expr.func {
        Expression::Ident(ident) => ident,
        _ => return None,
    };

    match ident.name.as_str() {
        // Special logic for handling the function `assume` from our Noir `fv_std` library
        s if s == ASSUME => {
            assert!(
                call_expr.arguments.len() == 1,
                "Expected function `assume` from `noir_fv_std` to have exactly one argument"
            );
            let condition_expr = ast_expr_to_vir_expr(&call_expr.arguments[0], Mode::Spec, globals);

            let exprx = ExprX::AssertAssume { is_assume: true, expr: condition_expr };

            let assume_expr = SpannedTyped::new(
                &build_span_no_id(
                    format!("Assume {} is true", call_expr.arguments[0]),
                    Some(call_expr.location),
                ),
                &make_unit_vir_type(),
                exprx,
            );

            Some(wrap_with_ghost_block(assume_expr, Some(call_expr.location)))
        }

        // Special logic for handling the function `old` from our Noir `fv_std` library
        s if s == OLD => {
            assert!(
                call_expr.arguments.len() == 1,
                "Expected function `old` from `noir_fv_std` to have exactly one argument"
            );

            let Expression::Ident(var_ident) = &call_expr.arguments[0] else {
                return None;
            };
            let ident_id =
                ast_definition_to_id(&var_ident.definition).expect("Definition doesn't have an id");
            let vir_ident = ast_ident_to_vir_var_ident(var_ident, ident_id);

            let exprx = ExprX::VarAt(vir_ident, vir::ast::VarAt::Pre);

            Some(SpannedTyped::new(
                &build_span_no_id(
                    format!("old({})", call_expr.arguments[0]),
                    Some(call_expr.location),
                ),
                &ast_type_to_vir_type(&call_expr.return_type),
                exprx,
            ))
        }

        _ => None,
    }
}
