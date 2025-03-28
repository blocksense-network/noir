use std::sync::Arc;

use im::HashSet;
use vir::ast::{Expr, ExprX, Exprs, Mode, SpannedTyped, Stmt};

use crate::ssa::verus_vir_gen::{
    build_span,
    expr_to_vir::{
        exprs::{
            get_enable_side_effects_value_id, instruction_to_expr, is_instruction_call_to_print,
        },
        patterns::instruction_to_stmt,
        types::get_function_ret_type,
    },
};

use super::{DataFlowGraph, Function, FvInstruction, Id, Instruction, SSAContext};

fn get_all_quantifier_indexes(
    attribute_instructions: &Vec<(Id<Instruction>, Instruction)>,
) -> HashSet<String> {
    let mut quantifier_indexes = HashSet::new();
    for (_, instruction) in attribute_instructions {
        if let Instruction::QuantStart { quant_type: _, indexes } = instruction {
            for index in indexes {
                quantifier_indexes.insert(index.clone());
            }
        }
    }

    quantifier_indexes
}

/// Converts annotations (aka attributes) to a vector of VIR expression.
fn func_attributes_to_vir_expr(
    attribute_instructions: Vec<(Id<Instruction>, Instruction)>,
    dfg: &DataFlowGraph,
    current_context: &mut SSAContext,
) -> Vec<Expr> {
    if let Some((last_instruction_id, last_instruction)) = attribute_instructions.last() {
        let mut vir_statements: Vec<Stmt> = Vec::new();
        let quantifier_indexes = get_all_quantifier_indexes(&attribute_instructions);

        for (instruction_id, instruction) in attribute_instructions.iter() {
            if is_instruction_call_to_print(&instruction_id, dfg) {
                continue;
            }
            match instruction {
                Instruction::Constrain(..)
                | Instruction::RangeCheck { .. }
                | Instruction::IncrementRc { .. } => continue,
                Instruction::Allocate { .. } => {
                    if dfg
                        .instruction_results(*instruction_id)
                        .iter()
                        .any(|val_id| quantifier_indexes.contains(&val_id.to_string()))
                    {
                        continue;
                    }
                }
                Instruction::QuantStart { quant_type: _, indexes } => {
                    mark_quantifier_start(indexes.clone(), current_context);
                    continue;
                }
                Instruction::EnableSideEffectsIf { .. } => {
                    current_context.side_effects_condition =
                        get_enable_side_effects_value_id(&instruction_id, dfg);
                    continue;
                }
                _ => {
                    let statement = instruction_to_stmt(
                        &instruction,
                        dfg,
                        *instruction_id,
                        Mode::Spec,
                        current_context,
                    );
                    if current_context.quantifier_context.is_inside_quantifier_body() {
                        current_context.quantifier_context.push_statement(statement);
                    } else {
                        vir_statements.push(statement);
                    }
                }
            }
        }

        let last_expr = if let Instruction::QuantEnd { .. } = last_instruction {
            match &(vir_statements.last()).expect("Attribute should not be empty").x {
                vir::ast::StmtX::Expr(expr) => expr.clone(),
                vir::ast::StmtX::Decl { pattern: _, mode: _, init } => {
                    init.clone().expect("Expected quant to be initialized")
                }
            }
        } else {
            instruction_to_expr(
                last_instruction_id.clone(),
                last_instruction,
                Mode::Spec,
                dfg,
                current_context,
            )
        };
        vec![SpannedTyped::new(
            &build_span(
                last_instruction_id,
                "Formal verification expression".to_string(),
                Some(dfg.get_call_stack(*last_instruction_id)),
            ),
            &get_function_ret_type(dfg.instruction_results(*last_instruction_id), dfg),
            ExprX::Block(Arc::new(vir_statements), Some(last_expr)),
        )]
    } else {
        vec![]
    }
}

fn mark_quantifier_start(indexes: Vec<String>, current_context: &mut SSAContext) {
    current_context.quantifier_context.start_quantifier(indexes);
}

/// For a given SSA function converts the `requires` annotations to VIR expressions. 
pub(crate) fn func_requires_to_vir_expr(
    func: &Function,
    current_context: &mut SSAContext,
) -> Exprs {
    let attr_instrs: Vec<(Id<Instruction>, Instruction)> = func
        .dfg
        .fv_instructions
        .iter()
        .enumerate()
        .map(|(ind, fv_instr)| (Id::new(ind + func.dfg.fv_start_id), fv_instr))
        .filter_map(|(ind, fv_instr)| {
            if let FvInstruction::Requires(instr) = &fv_instr {
                Some((ind, instr.clone()))
            } else {
                None
            }
        })
        .collect();
    Arc::new(func_attributes_to_vir_expr(attr_instrs, &func.dfg, current_context))
}

/// For a given SSA function converts the `ensures` annotations to VIR expressions. 
pub(crate) fn func_ensures_to_vir_expr(func: &Function, current_context: &mut SSAContext) -> Exprs {
    let attr_instrs: Vec<(Id<Instruction>, Instruction)> = func
        .dfg
        .fv_instructions
        .iter()
        .enumerate()
        .map(|(ind, fv_instr)| (Id::new(ind + func.dfg.fv_start_id), fv_instr))
        .filter_map(|(ind, fv_instr)| {
            if let FvInstruction::Ensures(instr) = &fv_instr {
                Some((ind, instr.clone()))
            } else {
                None
            }
        })
        .collect();
    Arc::new(func_attributes_to_vir_expr(attr_instrs, &func.dfg, current_context))
}
