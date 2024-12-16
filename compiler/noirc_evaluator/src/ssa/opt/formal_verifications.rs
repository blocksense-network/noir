use std::collections::HashMap;

use crate::ssa::ir::dfg::DataFlowGraph;
use crate::ssa::ir::function::{Function, FunctionId};
use crate::ssa::ir::instruction::{FvInstruction, Instruction, InstructionId};
use crate::ssa::ir::map::Id;
use crate::ssa::ir::value::{Value, ValueId};
use crate::ssa::ssa_gen::Ssa;

struct RightShiftInfo {
    pub start_index: usize,
    pub shift_amount: usize,
}

impl Ssa {
    pub(crate) fn formal_verifications_optimization(mut self) -> Self {
        // There is a loop bound to avoid infinite run.
        for _ in 0..1000 {
            if !self.has_call_instruction() {
                break;
            }
            self.inline_first_call();
        }
        self
    }

    fn has_call_instruction(&self) -> bool {
        let is_done = true;
        for function in self.functions.values() {
            let dfg = &function.dfg;
            for fv_instruction in dfg.fv_instructions.iter() {
                match fv_instruction {
                    FvInstruction::Requires(instruction) | FvInstruction::Ensures(instruction) => {
                        match instruction {
                            Instruction::Call { func: _, arguments: _ } => {
                                return true;
                            }
                            _ => {}
                        }
                    }
                }
            }
        }
        !is_done
    }

    pub(crate) fn inline_first_call(&mut self) {
        // Maps a function and the new values that will be added to its dfg.values.
        let mut update_func_with_values: Vec<(FunctionId, Vec<(usize, Value)>)> = Vec::new();

        // Maps a function and the new fv instructions which will overwrite the old ones.
        let mut update_func_fv_instructions: Vec<(FunctionId, Vec<FvInstruction>)> = Vec::new();

        // For each function that will be modified we have to right shift the fv instructions which will not be inlined
        // so we could make "space" for the ones which will be added by the inlining.
        let mut update_func_fv_right_shift: Vec<(FunctionId, Vec<RightShiftInfo>)> = Vec::new();

        // We also have to update the results map, because we need to update the indexes which
        let mut update_func_dfg_results: Vec<(FunctionId, Vec<(InstructionId, Vec<ValueId>)>)> =
            Vec::new();

        for function in self.functions.values() {
            let dfg = &function.dfg;
            let mut next_id = dfg
                .values_iter()
                .map(|(id, _)| id.to_usize())
                .max()
                .expect("Values can't be empty")
                + 1;
            let mut new_instructions: Vec<FvInstruction> = vec![];
            let mut index = 0;
            // First element in the tuple is the starting index for the right shift.
            // Second element is how much we have to right shift the instructions in dfg.results.
            let mut shift_fv_instructions: Vec<RightShiftInfo> = Vec::new();
            let mut current_needed_right_shifts = 0;
            let mut is_done = false;
            for fv_instruction in dfg.fv_instructions.iter() {
                let wrapper_function = match fv_instruction {
                    FvInstruction::Requires(_) => {
                            |instr| FvInstruction::Requires(instr)
                    }
                    FvInstruction::Ensures(_) => {
                            |instr| FvInstruction::Ensures(instr)
                    }
                };
                self.handle_fv_instruction(
                    function,
                    fv_instruction.give_inner_as_ref(),
                    index,
                    &mut is_done,
                    &mut new_instructions,
                    dfg,
                    &mut update_func_with_values,
                    &mut update_func_dfg_results,
                    &mut shift_fv_instructions,
                    &mut current_needed_right_shifts,
                    &mut next_id,
                    wrapper_function,
                );
                index += 1;
            }
            update_func_fv_instructions.push((function.id(), new_instructions));
            update_func_fv_right_shift.push((function.id(), shift_fv_instructions));
        }

        // Update the value map in the given function's dfg.
        for (func_id, values_to_create) in update_func_with_values {
            let function = self.functions.get_mut(&func_id).expect("No function should be missing");
            for (expected_val_id, value_to_insert) in values_to_create {
                let created_val_id = function.dfg.make_value(value_to_insert);
                assert!(created_val_id.to_usize() == expected_val_id);
            }
        }

        // Right shift all instructions which are after the call instruction which we are inlining.
        for (func_id, right_shift_info) in update_func_fv_right_shift {
            let function = self.functions.get_mut(&func_id).expect("No function should be missing");
            for right_shift_info in right_shift_info {
                let last_instruction_index =
                    function.dfg.fv_start_id + function.dfg.fv_instructions.len() - 1;

                for i in (right_shift_info.start_index..=last_instruction_index).rev() {
                    let instruction_id = InstructionId::new(i);
                    let new_instruction_id =
                        InstructionId::new(i + right_shift_info.shift_amount - 1);
                    let instr_results = function.dfg.instruction_results(instruction_id);
                    function
                        .dfg
                        .direct_insert_to_result(new_instruction_id, instr_results.to_vec());
                }
            }
        }

        // Insert all new instructions and their results in the result map of the given function's dfg.
        for (func_id, instructions_to_add) in update_func_dfg_results {
            let function = self.functions.get_mut(&func_id).expect("No function should be missing");
            for (instruction_id, value_ids) in instructions_to_add {
                function.dfg.direct_insert_to_result(instruction_id, value_ids);
            }
        }

        // Update the fv instruction vector for each function which has been modified.
        for (func_id, new_fv_instructions) in update_func_fv_instructions {
            let function = self.functions.get_mut(&func_id).expect("No function should be missing");
            function.dfg.fv_instructions = new_fv_instructions;
        }
    }

    /// Does inline expansion on the given call instruction.
    /// Updates the needed vectors with the newly inlined instructions.
    fn handle_fv_instruction<F>(
        &self,
        function: &Function,
        instruction: &Instruction,
        index: usize,
        is_done: &mut bool,
        new_instructions: &mut Vec<FvInstruction>,
        dfg: &DataFlowGraph,
        update_func_with_values: &mut Vec<(FunctionId, Vec<(usize, Value)>)>,
        update_func_dfg_results: &mut Vec<(FunctionId, Vec<(InstructionId, Vec<ValueId>)>)>,
        shift_fv_instructions: &mut Vec<RightShiftInfo>,
        current_needed_right_shifts: &mut usize,
        next_id: &mut usize,
        wrap_instruction: F,
    ) where
        F: Fn(Instruction) -> FvInstruction,
    {
        match instruction {
            Instruction::Call { func, arguments } => {
                if *is_done {
                    new_instructions.push(wrap_instruction(instruction.clone()))
                } else {
                    let instruction_id = InstructionId::new(dfg.fv_start_id + index);
                    let result_id = dfg.instruction_results(instruction_id);
                    let callee = match dfg[*func] {
                        Value::Function(inner_id) => {
                            self.functions.get(&inner_id).expect("Functions should have the id")
                        }
                        _ => unreachable!("You can only call functions"),
                    };
                    let (expanded_call_instructions, values_to_create, new_instr_ids) =
                        inline_expand(
                            arguments,
                            result_id.to_vec(),
                            callee,
                            instruction_id.to_usize(),
                            next_id,
                        );
                    *is_done = true;

                    update_func_with_values.push((function.id(), values_to_create));
                    shift_fv_instructions.push(RightShiftInfo {
                        start_index: instruction_id.to_usize() + *current_needed_right_shifts,
                        shift_amount: expanded_call_instructions.len(),
                    });
                    *current_needed_right_shifts += expanded_call_instructions.len();
                    update_func_dfg_results.push((function.id(), new_instr_ids));

                    new_instructions.extend(
                        expanded_call_instructions.into_iter().map(|instr| wrap_instruction(instr)),
                    );
                }
            }
            _ => new_instructions.push(wrap_instruction(instruction.clone())),
        }
    }
}

fn inline_expand(
    arguments: &Vec<ValueId>,
    result_ids: Vec<ValueId>,
    callee: &Function,
    starting_index: usize,
    next_id: &mut usize,
) -> (Vec<Instruction>, Vec<(usize, Value)>, Vec<(InstructionId, Vec<ValueId>)>) {
    let callee_dfg = &callee.dfg;
    let callee_entry_block = &callee_dfg[callee.entry_block()];
    let callee_parameters = callee_entry_block.parameters();

    let mut new_instructions: Vec<(InstructionId, Instruction)> = Vec::new();
    // Instruction ids and their returned values.
    let mut instr_id_to_returned_vals: Vec<(InstructionId, Vec<ValueId>)> = Vec::new();
    // Maps value ids in the callee to new value ids to be created for the caller.
    let mut value_id_map: HashMap<ValueId, ValueId> = HashMap::new();
    value_id_map.extend(callee_parameters.iter().zip(arguments.iter()));

    let mut values_to_create: Vec<(usize, Value)> = vec![];
    let mut current_instr_index = starting_index;

    for instruction_id in callee_entry_block.instructions() {
        let new_instruction = &callee_dfg[*instruction_id].map_values(|val_id| {
            map_value_id(val_id, callee_dfg, &mut values_to_create, &mut value_id_map, next_id)
        });
        new_instructions.push((instruction_id.clone(), new_instruction.clone()));
    }

    result_ids
        .iter()
        .zip(new_instructions[new_instructions.len().saturating_sub(result_ids.len())..].iter())
        .for_each(|(result_id, (instruction_id, _))| {
            value_id_map.insert(callee_dfg.instruction_results(*instruction_id)[0], *result_id);
        });

    for (instruction_id, _) in new_instructions.iter() {
        let instr_id = InstructionId::new(current_instr_index);
        instr_id_to_returned_vals.push((
            instr_id,
            callee_dfg
                .instruction_results(*instruction_id)
                .iter()
                .map(|val_id| {
                    value_id_map.get(&callee_dfg.resolve(*val_id)).unwrap_or(val_id).clone()
                })
                .collect(),
        ));
        current_instr_index += 1;
    }
    println!("instr ids and return vals{:?}", instr_id_to_returned_vals);

    (
        new_instructions.into_iter().map(|(_, instr)| instr).collect(),
        values_to_create,
        instr_id_to_returned_vals,
    )
}

/// If we have not already mapped the value_id to a new one we will create
/// a new value_id and add a new value to the `values_to_create` vector.
/// The new value is a copy of the older one but it has a new value_id.
fn map_value_id(
    lhs: ValueId,
    callee_dfg: &DataFlowGraph,
    values_to_create: &mut Vec<(usize, Value)>,
    value_id_map: &mut HashMap<ValueId, ValueId>,
    next_id: &mut usize,
) -> ValueId {
    match value_id_map.get(&lhs) {
        Some(mapped_value) => mapped_value.clone(),
        None => {
            let lhs_value = callee_dfg[lhs].clone();
            values_to_create.push((*next_id, lhs_value));
            let new_lhs: ValueId = Id::new(*next_id);
            value_id_map.insert(lhs, new_lhs);
            *next_id += 1;
            new_lhs
        }
    }
}
