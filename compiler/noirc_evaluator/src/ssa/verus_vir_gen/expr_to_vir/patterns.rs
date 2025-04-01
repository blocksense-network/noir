use expr_to_vir::{
    exprs::instruction_to_expr,
    types::{from_noir_type, get_function_ret_type},
};
use vir::ast::Binders;
use vir::ast_util::ident_binder;

use crate::ssa::verus_vir_gen::*;

use super::exprs::wrap_with_an_if_logic;

fn lhs_value_to_pattern(
    value_id: &ValueId,
    dfg: &DataFlowGraph,
    instruction_id: &InstructionId,
) -> Pattern {
    let value_patternx = PatternX::Var { name: id_into_var_ident(*value_id), mutable: false }; // Mutability not supported for the prototype
                                                                                               // Dereference type if we are initializing with allocate.
    let vir_type = if let Instruction::Allocate = dfg[*instruction_id] {
        match dfg[*value_id].get_type().clone() {
            Type::Reference(inner_type) => from_noir_type(inner_type.as_ref().clone(), None),
            _ => unreachable!(),
        }
    } else {
        from_noir_type(dfg[*value_id].get_type().clone(), None)
    };
    SpannedTyped::new(
        &build_span(
            value_id,
            format!("Lhs value({})", value_id),
            Some(dfg.get_value_call_stack(*value_id)),
        ),
        &vir_type,
        // &from_noir_type(dfg[*value_id].get_type().clone(), None),
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
                ident_binder(
                    &Arc::new(ind.to_string()),
                    &lhs_value_to_pattern(id, dfg, &instruction_id),
                )
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
            Some(dfg.get_call_stack(instruction_id)),
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
        1 => lhs_value_to_pattern(&lhs_ids[0], dfg, &instruction_id),
        _ => lhs_values_to_pattern(lhs_ids, dfg, instruction_id),
    }
}

/// Converts SSA instruction to a VIR statement.
pub(crate) fn instruction_to_stmt(
    instruction: &Instruction,
    dfg: &DataFlowGraph,
    instruction_id: Id<Instruction>,
    mode: Mode,
    current_context: &mut SSAContext,
) -> Stmt {
    let instruction_span = build_span(
        &instruction_id,
        format!("Instruction({}) statement", instruction_id),
        Some(dfg.get_call_stack(instruction_id)),
    );

    match dfg.instruction_results(instruction_id).len() {
        0 => {
            let instruction_as_expr =
                instruction_to_expr(instruction_id, instruction, mode, dfg, current_context);
            
            if let Some(condition_id) = current_context.side_effects_condition {
                // We have to wrap with an `if` our expression.
                let if_wrapped_expr = wrap_with_an_if_logic(
                    condition_id,
                    instruction_as_expr,
                    None,
                    dfg,
                    current_context.result_id_fixer,
                );
                Spanned::new(
                    build_span(
                        &instruction_id,
                        format!("Instruction({}) wrapped with if", instruction_id),
                        Some(dfg.get_call_stack(instruction_id)),
                    ),
                    StmtX::Expr(if_wrapped_expr),
                )
            } else {
                Spanned::new(
                    build_span(
                        &instruction_id,
                        format!("Instruction({})", instruction_id),
                        Some(dfg.get_call_stack(instruction_id)),
                    ),
                    StmtX::Expr(instruction_as_expr),
                )
            }
        }
        _ => Spanned::new(
            instruction_span,
            StmtX::Decl {
                pattern: instruction_to_pattern(instruction_id, dfg),
                mode: Some(mode),
                init: if let Instruction::Allocate = instruction {
                    None
                } else {
                    Some(instruction_to_expr(
                        instruction_id,
                        instruction,
                        mode,
                        dfg,
                        current_context,
                    ))
                },
            },
        ),
    }
}
