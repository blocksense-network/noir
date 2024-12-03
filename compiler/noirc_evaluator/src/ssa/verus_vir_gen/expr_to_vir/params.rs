use expr_to_vir::types::{from_noir_type, get_function_ret_type};
use function::get_function_return_values;
use vir::ast::{Mode, Param, Typ};

// use crate::ssa::verus_vir_gen::{function::get_function_return_values, ValueId};
use crate::ssa::verus_vir_gen::*;
/// Returns a Verus VIR param
fn build_param(
    value_id: ValueId,
    vir_type: Typ,
    mode: Mode,
    is_mut: bool,
    unwrapped_info: Option<(Mode, VarIdent)>,
    position: Option<usize>, // In some cases there is no way to indicate a position
    dfg: &DataFlowGraph,
) -> Param {
    let paramx = ParamX {
        name: id_into_var_ident(value_id),
        typ: vir_type,
        mode: mode,                     // For now all parameters are of type Exec
        is_mut: is_mut,                 // As far as I understand there is no &mut in SSA
        unwrapped_info: unwrapped_info, // Only if the parameter uses Ghost(x)/Tracked(x) pattern
    };
    let debug_string = if let Some(position_index) = position {
        "param position ".to_owned() + &position_index.to_string()
    } else {
        value_id.to_string()
    };
    Spanned::new(
        build_span(&value_id, debug_string, Some(dfg.get_value_call_stack(value_id))),
        paramx,
    )
}

fn build_tuple_return_param(
    values: &Vec<ValueId>,
    basic_block_id: Id<BasicBlock>,
    dfg: &DataFlowGraph,
) -> Param {
    if values.len() == 1 {
        return build_param(
            values[0],
            get_function_ret_type(values, dfg),
            Mode::Exec,
            false,
            None,
            None,
            dfg,
        );
    }

    let paramx = ParamX {
        name: VarIdent(Arc::new("result".to_string()), vir::ast::VarIdentDisambiguate::NoBodyParam),
        typ: get_function_ret_type(values, dfg),
        mode: Mode::Exec,
        is_mut: false,
        unwrapped_info: None,
    };
    Spanned::new(
        build_span(
            &basic_block_id,
            "Tuple param".to_string(),
            None,
            // Some(dfg.get_value_call_stack(*values.last().unwrap())),
        ),
        paramx,
    )
}

fn ssa_param_into_vir_param(
    value_id: ValueId,
    dfg: &DataFlowGraph,
) -> Result<Param, BuildingKrateError> {
    let value = dfg[value_id].clone();
    match value {
        Value::Param { block: _, position, typ } => {
            let vir_type = from_noir_type(typ, None);
            return Ok(build_param(
                value_id,
                vir_type,
                Mode::Exec,
                false,
                None,
                Some(position),
                dfg,
            ));
        }
        _ => {
            return Err(BuildingKrateError::SomeError(
                "expected SSA param value, found something else".to_string(),
            ))
        }
    };
}

pub(crate) fn get_function_return_param(func: &Function) -> Result<Param, BuildingKrateError> {
    let return_values = get_function_return_values(func)?;
    Ok(build_tuple_return_param(&return_values, func.entry_block(), &func.dfg))
}

pub(crate) fn get_function_params(func: &Function) -> Result<Params, BuildingKrateError> {
    let entry_block_id = func.entry_block();
    let entry_block = func.dfg[entry_block_id].clone();

    let mut parameters: Vec<Param> = Vec::new();
    for value_id in entry_block.parameters().iter() {
        let param = ssa_param_into_vir_param(*value_id, &func.dfg)?;
        parameters.push(param);
    }
    Ok(Arc::new(parameters))
}
