use std::sync::Arc;

use vir::ast::{
    Expr, FunctionAttrs, FunctionAttrsX, FunctionX, GenericBounds, ItemKind, Mode, Module,
    SpannedTyped, Visibility,
};

use super::{
    attributes::{func_ensures_to_vir_expr, func_requires_to_vir_expr},
    build_span, empty_vec_idents,
    expr_to_vir::{
        exprs::basic_block_to_exprx,
        params::{get_function_params, get_function_return_param},
    },
    func_id_into_funx_name, get_func_kind, BuildingKrateError, SSAContext, Function,
    FunctionId, ResultIdFixer, TerminatorInstruction, ValueId,
};

fn empty_vec_generic_bounds() -> GenericBounds {
    Arc::new(vec![])
}

fn is_function_return_void(func: &Function) -> bool {
    !func.returns().is_empty()
}

pub(crate) fn get_function_return_values(
    func: &Function,
) -> Result<Vec<ValueId>, BuildingKrateError> {
    let entry_block_id = func.entry_block();
    let terminating_instruction = func.dfg[entry_block_id].terminator();

    match terminating_instruction {
        Some(instruction) => match instruction {
            TerminatorInstruction::Return { return_values, call_stack: _ } => {
                let return_values: Vec<ValueId> =
                    return_values.iter().map(|val_id| func.dfg.resolve(*val_id)).collect();
                Ok(return_values)
            }
            _ => unreachable!(), // Only Brillig functions have a non Return Terminating instruction
        },
        None => {
            return Err(BuildingKrateError::SomeError(
                "Found None as a terminating instruction in a finished SSA block".to_string(),
            ))
        }
    }
}

/// Returns default instance of FunctionAttrs
/// By default we mean the same way a default instance would be
/// constructed in Verus VIR
fn build_default_funx_attrs(zero_args: bool) -> FunctionAttrs {
    Arc::new(FunctionAttrsX {
        uses_ghost_blocks: true,
        inline: false,
        hidden: Arc::new(vec![]), // Default in Verus
        broadcast_forall: false,
        broadcast_forall_only: false,
        no_auto_trigger: false,
        custom_req_err: None, // Can actually be used to report errors
        autospec: None,
        bit_vector: false, // Verify using bit vector theory?
        atomic: false,     // Maybe only ghost functions are atomic
        integer_ring: false,
        is_decrease_by: false,
        check_recommends: false,
        nonlinear: true,
        spinoff_prover: false,
        memoize: false,
        rlimit: None,
        print_zero_args: zero_args, // Has no default value
        print_as_method: false,
        prophecy_dependent: false,
        size_of_broadcast_proof: false,
        is_type_invariant_fn: false,
    })
}

fn func_body_to_vir_expr(func: &Function, current_context: &mut SSAContext) -> Expr {
    let (block_exprx, block_type) =
        basic_block_to_exprx(func.entry_block(), &func.dfg, current_context);
    SpannedTyped::new(
        &build_span(&func.id(), format!("Function's({}) basic block body", func.id())),
        &block_type,
        block_exprx,
    )
}

pub(crate) fn build_funx(
    func_id: FunctionId,
    func: &Function,
    current_module: Module,
) -> Result<FunctionX, BuildingKrateError> {
    let function_params = get_function_params(func)?;

    let ret = get_function_return_param(func)?;
    let result_id_fixer = ResultIdFixer::new(func, &ret).ok();
    let mut current_context =
        SSAContext { result_id_fixer: result_id_fixer.as_ref(), side_effects_condition: None };

    let funx: FunctionX = FunctionX {
        name: func_id_into_funx_name(func_id),
        proxy: None, // No clue. In Verus documentation it says "Proxy used to declare the spec of this function"
        kind: get_func_kind(func), // As far as I understand all functions in SSA are of FunctionKind::Static
        visibility: Visibility { restricted_to: None }, // None is for functions with public visibility. There is no information if the current function is public or private.
        owning_module: Some(current_module.x.path.clone()),
        mode: Mode::Exec, // Currently all functions are Exec. In the near future we will support ghost functions.
        fuel: 1, // In Verus' documentation it says that 1 means visible. I don't understand visible to what exactly
        typ_params: empty_vec_idents(), // There are no generics in SSA
        typ_bounds: empty_vec_generic_bounds(), // There are no generics in SSA
        params: function_params.clone(),
        ret,
        require: func_requires_to_vir_expr(func, &mut current_context),
        ensure: func_ensures_to_vir_expr(func, &mut current_context),
        decrease: Arc::new(vec![]), // No such feature in the prototype
        decrease_when: None,        // No such feature in the prototype
        decrease_by: None,          // No such feature in the prototype
        fndef_axioms: None,         // Not sure what it is
        mask_spec: None,            // Not sure what it is
        unwind_spec: None, // To be able to use functions from VSTD we need None on unwinding
        item_kind: ItemKind::Function,
        publish: None, // Only if we use None we pass Verus checks.
        attrs: build_default_funx_attrs(function_params.is_empty()),
        body: Some(func_body_to_vir_expr(func, &mut current_context)), // Functions in SSA always have a boyd
        extra_dependencies: vec![], // Not needed for the prototype
        ens_has_return: is_function_return_void(func), // Should be true if the function returns a value
        returns: None, // SSA functions (I believe) always return values and never expressions. They could also return zero values.
    };
    Ok(funx)
}
