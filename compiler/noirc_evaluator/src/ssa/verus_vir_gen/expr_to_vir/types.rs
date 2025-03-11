use crate::ssa::verus_vir_gen::*;

/// In Verus VIR to represent a "no type" you have to return an empty tuple
pub(crate) fn get_empty_vir_type() -> Typ {
    Arc::new(TypX::Datatype(Dt::Tuple(0), Arc::new(Vec::new()), Arc::new(Vec::new())))
}

pub(crate) fn get_integer_bit_width(numeric_type: NumericType) -> Option<IntegerTypeBitwidth> {
    match numeric_type {
        NumericType::Signed { bit_size: _ } => None, // Expected behavior in Verus VIR
        NumericType::Unsigned { bit_size } => Some(IntegerTypeBitwidth::Width(bit_size)),
        NumericType::NativeField => Some(IntegerTypeBitwidth::Width(FieldElement::max_num_bits())),
    }
}

pub(crate) fn get_int_range(numeric_type: NumericType) -> IntRange {
    match numeric_type {
        NumericType::Signed { bit_size } => IntRange::I(bit_size),
        NumericType::Unsigned { bit_size } => IntRange::U(bit_size),
        NumericType::NativeField => IntRange::I(FieldElement::max_num_bits()), // TODO(totel) Document mapping Noir Fields
    }
}

pub(crate) fn trunc_target_int_range(numeric_type: &NumericType, target_bit_size: u32) -> IntRange {
    match numeric_type {
        NumericType::Signed { bit_size: _ } => IntRange::I(target_bit_size),
        NumericType::Unsigned { bit_size: _ } => IntRange::U(target_bit_size),
        NumericType::NativeField => IntRange::I(target_bit_size),
    }
}

pub(crate) fn from_numeric_type(numeric_type: NumericType) -> Typ {
    match numeric_type {
        NumericType::Signed { bit_size } => Arc::new(TypX::Int(IntRange::I(bit_size))),
        NumericType::Unsigned { bit_size } => {
            if bit_size == 1 {
                Arc::new(TypX::Bool)
            } else {
                Arc::new(TypX::Int(IntRange::U(bit_size)))
            }
        }
        NumericType::NativeField => Arc::new(TypX::Int(IntRange::I(FieldElement::max_num_bits()))),
    }
}

pub(crate) fn into_vir_const_int(number: usize) -> Typ {
    Arc::new(TypX::ConstInt(BigInt::from(number)))
}

/// Maps the Noir type of the result of an instruction to Verus VIR type
pub(crate) fn instr_res_type_to_vir_type(
    instr_res_type: InstructionResultType,
    dfg: &DataFlowGraph,
) -> Typ {
    match instr_res_type {
        InstructionResultType::Operand(val_id) => {
            if is_function_type(&dfg[val_id]) {
                from_noir_type(dfg[val_id].get_type().clone(), Some(get_func_id(&dfg[val_id])))
            } else {
                from_noir_type(dfg[val_id].get_type().clone(), None)
            }
        }
        InstructionResultType::Known(noir_type) => from_noir_type(noir_type, None),
        InstructionResultType::None => get_empty_vir_type(),
        InstructionResultType::Unknown => unreachable!(), //TODO(totel/Kamen) See when it appears in SSA
    }
}

/// Maps a Noir composite type to either a Verus VIR Tuple type
/// or to a Verus VIR type, if the composite type is only one element
pub(crate) fn from_composite_type(composite_type: Arc<CompositeType>) -> Typ {
    let composite_types = (*composite_type.clone()).clone();
    if composite_types.len() == 1 {
        return from_noir_type(composite_types[0].clone(), None);
    } else {
        let typs: Vec<Typ> =
            composite_types.into_iter().map(|noir_type| from_noir_type(noir_type, None)).collect();
        return Arc::new(TypX::Datatype(
            Dt::Tuple(typs.len()),
            Arc::new(typs),
            Arc::new(Vec::new()),
        ));
    }
}

/// Maps a Noir type to a Verus VIR type.
/// The `func_id` should be available when the Noir type is Function
pub(crate) fn from_noir_type(noir_typ: Type, func_id: Option<FunctionId>) -> Typ {
    match noir_typ {
        Type::Numeric(numeric_type) => from_numeric_type(numeric_type),
        Type::Reference(referenced_type) => Arc::new(TypX::Decorate(
            TypDecoration::Ref,
            None,
            from_noir_type((*referenced_type.clone()).clone(), None),
        )),
        Type::Array(composite_type, size) => Arc::new(TypX::Primitive(
            Primitive::Array,
            Arc::new(vec![from_composite_type(composite_type), into_vir_const_int(size)]),
        )),
        Type::Slice(composite_type) => Arc::new(TypX::Primitive(
            Primitive::Slice,
            Arc::new(vec![from_composite_type(composite_type)]),
        )),
        Type::Function => Arc::new(TypX::FnDef(
            func_id_into_funx_name(func_id.unwrap_or_else(|| {
                panic!("Unexpected lack of function id when type::function was met")
            })),
            Arc::new(Vec::new()),
            None,
        )),
    }
}

pub(crate) fn build_tuple_type(values: &Vec<ValueId>, dfg: &DataFlowGraph) -> Typ {
    let datatype: Dt = Dt::Tuple(values.len());
    let tuple_types: Typs = Arc::new(
        values.iter().map(|val_id| from_noir_type(dfg[*val_id].get_type().clone(), None)).collect(),
    );
    Arc::new(TypX::Datatype(datatype, tuple_types, Arc::new(vec![])))
}

pub(crate) fn get_function_ret_type(results: &[Id<Value>], dfg: &DataFlowGraph) -> Typ {
    match results.len() {
        0 => get_empty_vir_type(),
        1 => { // Dereference return type
            let noir_type = if let Type::Reference(inner_type) = dfg[results[0]].get_type() {
                inner_type.as_ref().clone()
            } else {
                dfg[results[0]].get_type().clone()
            };
            from_noir_type(noir_type, None)
        },
        _ => build_tuple_type(&results.to_vec(), dfg),
    }
}
