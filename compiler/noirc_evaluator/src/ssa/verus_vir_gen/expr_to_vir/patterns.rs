use expr_to_vir::{
    exprs::instruction_to_expr,
    types::{from_noir_type, get_function_ret_type},
};
use vir::ast::Binders;
use vir::ast_util::ident_binder;

use crate::ssa::verus_vir_gen::*;

fn lhs_value_to_pattern(value_id: &ValueId, dfg: &DataFlowGraph) -> Pattern {
    let value_patternx = PatternX::Var { name: id_into_var_ident(*value_id), mutable: false }; // Mutability not supported for the prototype
    SpannedTyped::new(
        &build_span(value_id, format!("Lhs value({})", value_id)),
        &from_noir_type(dfg[*value_id].get_type().clone(), None),
        value_patternx,
    )
}

/// Perhaps needed for tuples. Must be tested explicitly
fn lhs_values_to_pattern(
    lhs_values: &[Id<Value>],
    dfg: &DataFlowGraph,
    instruction_id: Id<Instruction>,
) -> Pattern {
    let tuple_count = lhs_values.len();
    let binders: Binders<Pattern> = Arc::new(
        lhs_values
            .iter()
            .enumerate()
            .map(|(ind, id)| {
                ident_binder(&Arc::new(ind.to_string()), &lhs_value_to_pattern(id, dfg))
            })
            .collect(),
    );
    let tuple_patternx =
        PatternX::Constructor(Dt::Tuple(tuple_count), prefix_tuple_variant(tuple_count), binders);
    SpannedTyped::new(
        &build_span(
            &instruction_id,
            format!(
                "Instruction({}) lhs values[{}]",
                instruction_id,
                lhs_values.iter().map(|x| x.to_string()).collect::<Vec<String>>().join(", ")
            ),
        ),
        &get_function_ret_type(lhs_values, dfg),
        tuple_patternx,
    )
}

/// Fetch the instruction results from the dfg. Those results are the
/// left hand side of the SSA. This can be observed in the function
/// `display_instructions()` in the printer.rs file.
fn instruction_to_pattern(instruction_id: InstructionId, dfg: &DataFlowGraph) -> Pattern {
    let lhs_ids = dfg.instruction_results(instruction_id);
    match lhs_ids.len() {
        0 => panic!("Instructions with no results can not be turned into a pattern"),
        1 => lhs_value_to_pattern(&lhs_ids[0], dfg),
        _ => lhs_values_to_pattern(lhs_ids, dfg, instruction_id),
    }
}

pub(crate) fn instruction_to_stmt(
    instruction: &Instruction,
    dfg: &DataFlowGraph,
    instruction_id: Id<Instruction>,
    mode: Mode,
    current_context: &mut SSAContext,
) -> Stmt {
    let instruction_span =
        build_span(&instruction_id, format!("Instruction({}) statement", instruction_id));

    match dfg.instruction_results(instruction_id).len() {
        0 => Spanned::new(
            build_span(&instruction_id, format!("Instruction({})", instruction_id)),
            StmtX::Expr(instruction_to_expr(
                instruction_id,
                instruction,
                mode,
                dfg,
                current_context,
            )),
        ),
        _ => Spanned::new(
            instruction_span,
            StmtX::Decl {
                pattern: instruction_to_pattern(instruction_id, dfg),
                mode: Some(mode),
                init: Some(instruction_to_expr(
                    instruction_id,
                    instruction,
                    mode,
                    dfg,
                    current_context,
                )),
            },
        ),
    }
}
