use super::expr_to_vir::expr::func_body_to_vir_expr;
use super::{
    BuildingKrateError,
    attribute::{func_ensures_to_vir_expr, func_requires_to_vir_expr},
};
use noirc_frontend::monomorphization::ast::Function;
use std::sync::Arc;
use vir::ast::{
    BodyVisibility, Fun, FunctionAttrs, FunctionKind, FunctionX, ItemKind, Mode, Module,
    Opaqueness, Param, Params, Visibility,
};

fn function_into_funx_name(function: &Function) -> Fun {
    todo!()
}

fn is_ghost_function(function: &Function) -> bool {
    todo!()
}

fn get_function_opaqueness(is_ghost: bool) -> Opaqueness {
    todo!()
}

fn get_function_mode(is_ghost: bool) -> Mode {
    todo!()
}

fn get_function_params(function: &Function) -> Result<Params, BuildingKrateError> {
    todo!()
}

fn get_function_return_param(function: &Function) -> Result<Param, BuildingKrateError> {
    todo!()
}

fn is_function_return_void(func: &Function) -> bool {
    todo!()
}

fn build_default_funx_attrs(zero_args: bool) -> FunctionAttrs {
    todo!()
}

// Converts the given Monomorphized AST function into a VIR function.
pub fn build_funx(
    function: &Function,
    current_module: &Module,
) -> Result<FunctionX, BuildingKrateError> {
    let is_ghost = is_ghost_function(function);
    let function_params = get_function_params(function)?;
    let function_return_param = get_function_return_param(function)?;

    let funx = FunctionX {
        name: function_into_funx_name(function),
        proxy: None, // Only needed for external fn specifications which we currently don't support
        kind: FunctionKind::Static, // Monomorphized AST has only static functions
        visibility: Visibility {
            restricted_to: None, // `None` is for functions with public visibility
        }, // Categorization for public/private visibility doesn't exist in the Mon. AST
        body_visibility: BodyVisibility::Visibility(Visibility {
            restricted_to: None, // We currently support only fully visible ghost functions
        }),
        opaqueness: get_function_opaqueness(is_ghost),
        owning_module: Some(current_module.x.path.clone()), // The module in which this function is located.
        mode: get_function_mode(is_ghost),
        typ_params: Arc::new(Vec::new()), // There are no generics in Monomorphized AST
        typ_bounds: Arc::new(Vec::new()), // There are no generics in Monomorphized AST
        params: function_params,
        ret: function_return_param,
        ens_has_return: is_function_return_void(function),
        require: func_requires_to_vir_expr(function),
        ensure: func_ensures_to_vir_expr(function),
        returns: None, // We don't support the special clause called `return`
        decrease: Arc::new(vec![]), // Annotation for recursive functions. We currently don't support it
        decrease_when: None, // Annotation for recursive functions. We currently don't support it
        decrease_by: None,   // Annotation for recursive functions. We currently don't support it
        fndef_axioms: None, // We currently don't support this feature
        mask_spec: None, // We currently don't support this feature
        unwind_spec: None, // To be able to use functions from Verus std we need None on unwinding
        item_kind: ItemKind::Function,
        attrs: build_default_funx_attrs(function.parameters.is_empty()),
        body: Some(func_body_to_vir_expr(function)),
        extra_dependencies: Vec::new(),
    };

    Ok(funx)
}
