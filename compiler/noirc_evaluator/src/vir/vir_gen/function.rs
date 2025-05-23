use crate::vir::vir_gen::build_span_no_id;
use crate::vir::vir_gen::expr_to_vir::types::ast_type_to_vir_type;

use super::expr_to_vir::expr::func_body_to_vir_expr;
use super::expr_to_vir::params::ast_param_to_vir_param;
use super::{
    BuildingKrateError,
    attribute::{func_ensures_to_vir_expr, func_requires_to_vir_expr},
};
use noirc_errors::Location;
use noirc_frontend::monomorphization::ast::{Function, MonomorphizedFvAttribute, Type};
use std::sync::Arc;
use vir::ast::{
    BodyVisibility, Fun, FunX, FunctionAttrs, FunctionAttrsX, FunctionKind, FunctionX, ItemKind,
    Mode, Module, Opaqueness, Param, ParamX, Params, PathX, VarIdent, VarIdentDisambiguate,
    Visibility,
};
use vir::def::Spanned;

fn function_into_funx_name(function: &Function) -> Fun {
    Arc::new(FunX {
        path: Arc::new(PathX {
            krate: None,
            segments: Arc::new(vec![Arc::new(function.id.to_string())]),
        }),
    })
}

fn is_ghost_function(function: &Function) -> bool {
    function
        .formal_verification_attributes
        .iter()
        .any(|x| matches!(x, MonomorphizedFvAttribute::Ghost))
}

fn get_function_mode(is_ghost: bool) -> Mode {
    if is_ghost { Mode::Spec } else { Mode::Exec }
}

fn get_function_params(function: &Function, mode: Mode) -> Result<Params, BuildingKrateError> {
    let locations: Vec<Location> =
        function.func_sig.0.iter().map(|param| param.0.location()).collect();
    let params_as_vir: Vec<Param> = function
        .parameters
        .iter()
        .zip(locations.into_iter())
        .map(|(param, location)| ast_param_to_vir_param(param, location, mode, &function.name))
        .collect();

    Ok(Arc::new(params_as_vir))
}

fn get_function_return_param(function: &Function, mode: Mode) -> Result<Param, BuildingKrateError> {
    let paramx = ParamX {
        name: VarIdent(Arc::new("result".to_string()), VarIdentDisambiguate::NoBodyParam),
        typ: ast_type_to_vir_type(&function.return_type),
        mode: mode,
        is_mut: false,
        unwrapped_info: None,
    };

    Ok(Spanned::new(
        build_span_no_id(format!("Result value of function {}", function.name), None),
        paramx,
    ))
}

fn is_function_return_void(function: &Function) -> bool {
    matches!(function.return_type, Type::Unit)
}

/// Returns default instance of FunctionAttrs.
/// By default we mean the same way a default instance would be
/// constructed in Verus during the phase Rust HIR -> VIR.
fn build_default_funx_attrs(zero_args: bool, is_constrained: bool) -> FunctionAttrs {
    Arc::new(FunctionAttrsX {
        uses_ghost_blocks: true,
        inline: false,
        hidden: Arc::new(vec![]), // Default in Verus
        broadcast_forall: false,
        broadcast_forall_only: false,
        no_auto_trigger: false,
        custom_req_err: None,
        autospec: None,
        bit_vector: false,
        atomic: false, //TODO(totel) Maybe ghost functions have to be defined as atomic
        integer_ring: false,
        is_decrease_by: false,
        check_recommends: false,
        nonlinear: true, // This flag was set specifically by us to support arithmetic multiplication.
        spinoff_prover: false,
        memoize: false,
        rlimit: None,
        print_zero_args: zero_args, // Has no default value
        print_as_method: false,
        prophecy_dependent: false,
        size_of_broadcast_proof: false,
        is_type_invariant_fn: false,
        auto_ext_equal: vir::ast::AutoExtEqual::default(),
        is_external_body: false, // Currently we don't support external_fn_specification
        is_unsafe: false,
        exec_assume_termination: is_constrained, // Constrained functions are practically total
        exec_allows_no_decreases_clause: false,
    })
}

// Converts the given Monomorphized AST function into a VIR function.
pub fn build_funx(
    function: &Function,
    current_module: &Module,
) -> Result<FunctionX, BuildingKrateError> {
    let is_ghost = is_ghost_function(function);
    let mode = get_function_mode(is_ghost);

    let function_params = get_function_params(function, mode)?;
    let function_return_param = get_function_return_param(function, mode)?;

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
        opaqueness: Opaqueness::Revealed {
            visibility: Visibility { restricted_to: None }, // We currently don't support opaqueness control
        },
        owning_module: Some(current_module.x.path.clone()), // The module in which this function is located.
        mode,
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
        fndef_axioms: None,  // We currently don't support this feature
        mask_spec: None,     // We currently don't support this feature
        unwind_spec: None,   // To be able to use functions from Verus std we need None on unwinding
        item_kind: ItemKind::Function,
        attrs: build_default_funx_attrs(function.parameters.is_empty(), !function.unconstrained),
        body: Some(func_body_to_vir_expr(function)),
        extra_dependencies: Vec::new(),
    };

    Ok(funx)
}
