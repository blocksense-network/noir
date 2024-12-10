use std::collections::HashMap;

use crate::ssa::ir::dfg::DataFlowGraph;
use crate::ssa::ir::function::Function;
use crate::ssa::ir::instruction::Binary;
use crate::ssa::ir::instruction::FvInstruction;
use crate::ssa::ir::instruction::Instruction;
use crate::ssa::ir::instruction::InstructionId;
use crate::ssa::ir::map::Id;
use crate::ssa::ir::value::Value;
use crate::ssa::ssa_gen::Ssa;

impl Ssa {
    pub(crate) fn formal_verifications_optimization(mut self) -> Self {
        for function in self.functions.values() {
            let dfg = &function.dfg;
            let mut next_id =
                dfg.values_iter().map(|(id, _)| id.to_usize()).max().expect("There should max") + 1;
            let mut new_instructions = vec![];
            let mut index = 0;
            for fv_instruction in dfg.fv_instructions.iter() {
                // println!("hi stanm [{:?}]", fv_instruction);
                if let FvInstruction::Ensures(instruction) = fv_instruction {
                    match instruction {
                        Instruction::Call { func, arguments } => {
                            let instruction_id = InstructionId::new(dfg.fv_start_id + index);
                            let result_id = dfg.instruction_results(instruction_id)[0];
                            let callee = match dfg[*func] {
                                Value::Function(inner_id) => self
                                    .functions
                                    .get(&inner_id)
                                    .expect("Functions should have the id"),
                                _ => panic!("at the disco"),
                            };
                            let expanded_call_instructions =
                                inline_expand(dfg, arguments, result_id, callee, &mut next_id);

                            new_instructions.extend(expanded_call_instructions);

                            // println!("hi stanm result id [{:?}]", result_id);
                        }
                        _ => new_instructions.push(instruction.clone()),
                    };
                }
                index += 1;
            }

            // dfg.fv_instructions = new_instructions;
        }

        self
    }
}

fn inline_expand(
    caller_dfg: &DataFlowGraph,
    arguments: &Vec<Id<Value>>,
    result_id: Id<Value>,
    callee: &Function,
    next_id: &mut usize,
) -> Vec<Instruction> {
    let callee_dfg = &callee.dfg;
    let callee_entry_block = &callee_dfg[callee.entry_block()];
    let callee_parameters = callee_entry_block.parameters();

    // maps value ids in the callee to new value ids to be created for the caller
    let mut value_id_map: HashMap<Id<Value>, Id<Value>> = HashMap::new();
    value_id_map.extend(callee_parameters.iter().zip(arguments.iter()));

    let mut values_to_create = vec![];

    for instruction_id in callee_entry_block.instructions() {
        let new_instruction = match &callee_dfg[*instruction_id] {
            Instruction::Binary(Binary { lhs, rhs, operator }) => {
                let new_lhs = map_value_id(
                    *lhs,
                    callee_dfg,
                    &mut values_to_create,
                    &mut value_id_map,
                    next_id,
                );

                let new_rhs = map_value_id(
                    *rhs,
                    callee_dfg,
                    &mut values_to_create,
                    &mut value_id_map,
                    next_id,
                );

                Instruction::binary(*operator, new_lhs, new_rhs)
            }
            Instruction::Cast(value_id, target_type) => {
                let new_value_id = map_value_id(
                    *value_id,
                    callee_dfg,
                    &mut values_to_create,
                    &mut value_id_map,
                    next_id,
                );

                Instruction::Cast(new_value_id, target_type.clone())
            }
            Instruction::Not(value_id) => {
                let new_value_id = map_value_id(
                    *value_id,
                    callee_dfg,
                    &mut values_to_create,
                    &mut value_id_map,
                    next_id,
                );

                Instruction::Not(new_value_id)
            }
            Instruction::Truncate { value, bit_size, max_bit_size } => todo!(),
            Instruction::Constrain(_, _, _) => todo!(),
            Instruction::RangeCheck { value, max_bit_size, assert_message } => todo!(),
            Instruction::Call { func, arguments } => todo!(),
            Instruction::Allocate => todo!(),
            Instruction::Load { address } => {
                let new_address = map_value_id(
                    *address,
                    callee_dfg,
                    &mut values_to_create,
                    &mut value_id_map,
                    next_id,
                );

                Instruction::Load { address: new_address }
            }
            Instruction::Store { address, value } => todo!(),
            Instruction::EnableSideEffectsIf { condition } => {
                let new_condition = map_value_id(
                    *condition,
                    callee_dfg,
                    &mut values_to_create,
                    &mut value_id_map,
                    next_id,
                );

                Instruction::EnableSideEffectsIf { condition: new_condition }
            }
            Instruction::ArrayGet { array, index } => todo!(),
            Instruction::ArraySet { array, index, value, mutable } => todo!(),
            Instruction::IncrementRc { value } => todo!(),
            Instruction::DecrementRc { value } => todo!(),
            Instruction::IfElse { then_condition, then_value, else_condition, else_value } => {
                todo!()
            }
        };

        println!("hi stanm inlining [{:?}]", new_instruction);
    }
    vec![]
}

fn map_value_id(
    lhs: Id<Value>,
    callee_dfg: &DataFlowGraph,
    values_to_create: &mut Vec<(usize, Value)>,
    value_id_map: &mut HashMap<Id<Value>, Id<Value>>,
    next_id: &mut usize,
) -> Id<Value> {
    match value_id_map.get(&lhs) {
        Some(mapped_value) => mapped_value.clone(),
        None => {
            let lhs_value = callee_dfg[lhs].clone();
            values_to_create.push((*next_id, lhs_value));
            let new_lhs: Id<Value> = Id::new(*next_id);
            value_id_map.insert(lhs, new_lhs);
            *next_id += 1;
            new_lhs
        }
    };

    if let Some(mapped_value) = value_id_map.get(&lhs) {
        mapped_value.clone()
    } else {
        let lhs_value = callee_dfg[lhs].clone();
        values_to_create.push((*next_id, lhs_value));
        let new_lhs: Id<Value> = Id::new(*next_id);
        value_id_map.insert(lhs, new_lhs);
        *next_id += 1;
        new_lhs
    }
}
