use std::sync::Arc;

use vir::ast::{Expr, ExprX, Exprs, Mode, SpannedTyped, Stmt};

use crate::ssa::verus_vir_gen::{
    build_span,
    expr_to_vir::{
        exprs::instruction_to_expr, patterns::instruction_to_stmt, types::get_function_ret_type,
    },
};

use super::{SSAContext, DataFlowGraph, Function, FvInstruction, Id, Instruction};

fn func_attributes_to_vir_expr(
    attribute_instructions: Vec<(Id<Instruction>, Instruction)>,
    dfg: &DataFlowGraph,
    current_context: &mut SSAContext,
) -> Vec<Expr> {
    if let Some((last_instruction_id, last_instruction)) = attribute_instructions.last() {
        let mut vir_statements: Vec<Stmt> = Vec::new();

        for (instruction_id, instruction) in attribute_instructions.clone() {
            if let Instruction::Constrain(_, _, _) = instruction {
                continue;
            }
            if let Instruction::RangeCheck { .. } = instruction {
                continue;
            }
            let statement =
                instruction_to_stmt(&instruction, dfg, instruction_id, Mode::Spec, current_context);
            vir_statements.push(statement);
        }

        let last_expr = instruction_to_expr(
            last_instruction_id.clone(),
            last_instruction,
            Mode::Spec,
            dfg,
            current_context,
        );
        vec![SpannedTyped::new(
            &build_span(last_instruction_id, "Formal verification expression".to_string()),
            &get_function_ret_type(dfg.instruction_results(*last_instruction_id), dfg),
            ExprX::Block(Arc::new(vir_statements), Some(last_expr)),
        )]
    } else {
        vec![]
    }
}

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

pub(crate) fn func_ensures_to_vir_expr(
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
            if let FvInstruction::Ensures(instr) = &fv_instr {
                Some((ind, instr.clone()))
            } else {
                None
            }
        })
        .collect();
    Arc::new(func_attributes_to_vir_expr(attr_instrs, &func.dfg, current_context))
}