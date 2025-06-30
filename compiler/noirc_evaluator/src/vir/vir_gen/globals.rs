use std::{collections::BTreeMap, sync::Arc};

use noirc_frontend::monomorphization::ast::{Expression, GlobalId, Type};
use vir::{
    ast::{
        BodyVisibility, Fun, FunX, FunctionAttrs, FunctionAttrsX, FunctionKind, FunctionX,
        ItemKind, Mode, Module, Opaqueness, Param, ParamX, PathX, VarIdent, VarIdentDisambiguate,
        Visibility,
    },
    def::Spanned,
};

use crate::vir::vir_gen::{
    build_span_no_id,
    expr_to_vir::{expr::ast_expr_to_vir_expr, types::ast_type_to_vir_type},
};

pub fn build_global_const_x(
    name: &str,
    ast_type: &Type,
    expression: &Expression,
    current_module: &Module,
    globals: &BTreeMap<GlobalId, (String, Type, Expression)>,
) -> FunctionX {
    let mode = Mode::Spec;

    let const_return_param = build_function_return_param(ast_type, name);

    let const_x = FunctionX {
        name: build_fun_from_name(name.to_string()),
        proxy: None, // Only needed for external fn specifications which we currently don't support
        kind: FunctionKind::Static, // Consts are of kind Static
        visibility: Visibility {
            restricted_to: None, // `None` is for public visibility
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
        params: Arc::new(Vec::new()),     // Constants don't have parameters
        ret: const_return_param,
        ens_has_return: true,
        require: Arc::new(Vec::new()),
        ensure: Arc::new(Vec::new()),
        returns: None, // We don't support the special clause called `return`
        decrease: Arc::new(vec![]), // Annotation for recursive functions. We currently don't support it
        decrease_when: None, // Annotation for recursive functions. We currently don't support it
        decrease_by: None,   // Annotation for recursive functions. We currently don't support it
        fndef_axioms: None,  // We currently don't support this feature
        mask_spec: None,     // We currently don't support this feature
        unwind_spec: None,   // To be able to use functions from Verus std we need None on unwinding
        item_kind: ItemKind::Const,
        attrs: build_default_constx_attrs(),
        body: Some(ast_expr_to_vir_expr(expression, mode, globals)),
        extra_dependencies: Vec::new(),
    };

    const_x
}

/// Uses the default name `%return` for function return parameter
fn build_function_return_param(return_type: &Type, const_name: &str) -> Param {
    let paramx = ParamX {
        name: VarIdent(Arc::new("%return".to_string()), VarIdentDisambiguate::AirLocal),
        typ: ast_type_to_vir_type(return_type),
        mode: Mode::Exec,
        is_mut: false,
        unwrapped_info: None,
    };

    Spanned::new(build_span_no_id(format!("Result value of constant {}", const_name), None), paramx)
}

fn build_fun_from_name(name: String) -> Fun {
    Arc::new(FunX {
        path: Arc::new(PathX { krate: None, segments: Arc::new(vec![Arc::new(name)]) }),
    })
}

fn build_default_constx_attrs() -> FunctionAttrs {
    Arc::new(FunctionAttrsX {
        uses_ghost_blocks: true,
        inline: false,
        hidden: Arc::new(vec![]),
        broadcast_forall: false,
        broadcast_forall_only: false,
        no_auto_trigger: false,
        custom_req_err: None,
        autospec: None,
        bit_vector: false,
        atomic: false,
        integer_ring: false,
        is_decrease_by: false,
        check_recommends: false,
        nonlinear: true,
        spinoff_prover: false,
        memoize: false,
        rlimit: None,
        print_zero_args: false,
        print_as_method: false,
        prophecy_dependent: false,
        size_of_broadcast_proof: false,
        is_type_invariant_fn: false,
        auto_ext_equal: vir::ast::AutoExtEqual::default(),
        is_external_body: false,
        is_unsafe: false,
        exec_assume_termination: false,
        exec_allows_no_decreases_clause: false,
    })
}
