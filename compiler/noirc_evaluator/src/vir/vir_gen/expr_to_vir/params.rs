use noirc_errors::Location;
use noirc_frontend::{
    monomorphization::ast::{LocalId, Type},
    shared::Visibility,
};
use vir::{
    ast::{Mode, Param, ParamX},
    def::Spanned,
};

use crate::vir::vir_gen::{
    build_span_no_id,
    expr_to_vir::{ast_var_into_var_ident, types::ast_type_to_vir_type},
};

type AstParam = (LocalId, /*mutable:*/ bool, /*name:*/ String, Type, Visibility);

pub fn ast_param_to_vir_param(
    parameter: &AstParam,
    location: Location,
    mode: Mode,
    function_name: &str,
) -> Param {
    let paramx = ParamX {
        name: ast_var_into_var_ident(parameter.2.clone(), parameter.0.0),
        typ: ast_type_to_vir_type(&parameter.3),
        mode,
        is_mut: parameter.1,
        unwrapped_info: None, // Special unwrapping pattern which we don't support
    };

    Spanned::new(
        build_span_no_id(format!("Parameters of the function {}", function_name), Some(location)),
        paramx,
    )
}
