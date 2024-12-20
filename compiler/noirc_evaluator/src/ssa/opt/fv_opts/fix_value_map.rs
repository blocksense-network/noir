use crate::ssa::{
    ir::{
        dfg::DataFlowGraph,
        instruction::InstructionId,
        types::Type,
        value::{Value, ValueId},
    },
    ssa_gen::Ssa,
};

impl Ssa {
    pub(crate) fn update_value_map(mut self) -> Self {
        let mut values_to_update: Vec<(Vec<ValueId>, Vec<Value>)> = Vec::new();
        for (_, function) in &mut self.functions {
            values_to_update.clear();

            for (fv_instruction_id, _) in function.dfg.get_fv_instructions_with_ids() {
                let values_ids = function.dfg.instruction_results(fv_instruction_id);

                let instr_as_values = transform_instruction_to_value(
                    fv_instruction_id,
                    values_ids.len(),
                    &function.dfg,
                );

                values_to_update.push((values_ids.to_vec(), instr_as_values));
            }

            for (values_ids, instructions) in values_to_update.iter() {
                values_ids.iter().zip(instructions.iter()).for_each(|(value_id, instruction)| {
                    function.dfg.update_value_at_id(*value_id, instruction.clone());
                });
            }
        }
        self
    }
}

fn transform_instruction_to_value(
    instruction_id: InstructionId,
    number_of_return_values: usize,
    dfg: &DataFlowGraph,
) -> Vec<Value> {
    let instruction_types: Vec<Type> = dfg
        .instruction_results(instruction_id)
        .iter()
        .map(|val_id| dfg[*val_id].get_type().clone())
        .collect();

    let mut instruction_as_values: Vec<Value> = Vec::new();

    for i in 0..number_of_return_values {
        instruction_as_values.push(Value::Instruction {
            instruction: instruction_id,
            position: i,
            typ: instruction_types[i].clone(),
        });
    }

    instruction_as_values
}
