use crate::ssa::verus_vir_gen::*;
use expr_to_vir::{
    patterns::instruction_to_stmt,
    types::{
        from_composite_type, from_noir_type, get_empty_vir_type, get_function_ret_type,
        get_int_range, get_integer_bit_width, instr_res_type_to_vir_type, into_vir_const_int,
        trunc_target_int_range,
    },
};
use noirc_frontend::ast::QuantifierType;
use num_bigint::ToBigInt;
use vir::ast::{
    AirQuant, Binder, BinderX, FieldOpr, Quant, TriggerAnnotation, VarBinder, VarBinderX,
};

fn get_value_bitwidth(value_id: &ValueId, dfg: &DataFlowGraph) -> IntegerTypeBitwidth {
    let value = &dfg[*value_id];
    match value.get_type() {
        Type::Numeric(numeric_type) => get_integer_bit_width(*numeric_type).unwrap(),
        _ => panic!("Bitwise operation on a non numeric type"),
    }
}

fn wrap_with_an_if_logic(
    condition_id: ValueId,
    binary_expr: Expr,
    lhs_expr: Expr,
    dfg: &DataFlowGraph,
    result_id_fixer: Option<&ResultIdFixer>,
) -> Expr {
    let lhs_type = lhs_expr.typ.clone();
    let if_exprx = ExprX::If(
        ssa_value_to_expr(&condition_id, dfg, result_id_fixer),
        binary_expr,
        Some(lhs_expr),
    );
    SpannedTyped::new(
        &build_span(
            &condition_id,
            format!("Enable side effects if"),
            Some(dfg.get_value_call_stack(condition_id)),
        ),
        &lhs_type,
        if_exprx,
    )
}

fn binary_op_to_vir_binary_op(
    binary: &BinaryOp,
    mode: Mode,
    lhs: &ValueId,
    dfg: &DataFlowGraph,
) -> VirBinaryOp {
    match binary {
        BinaryOp::Add => VirBinaryOp::Arith(ArithOp::Add, mode), // It would be of Mode::Spec only if it is a part of a fv attribute or a ghost block
        BinaryOp::Sub => VirBinaryOp::Arith(ArithOp::Sub, mode),
        BinaryOp::Mul => VirBinaryOp::Arith(ArithOp::Mul, mode),
        BinaryOp::Div => VirBinaryOp::Arith(ArithOp::EuclideanDiv, mode),
        BinaryOp::Mod => VirBinaryOp::Arith(ArithOp::EuclideanMod, mode),
        BinaryOp::Eq => VirBinaryOp::Eq(mode),
        BinaryOp::Lt => VirBinaryOp::Inequality(InequalityOp::Lt),
        BinaryOp::And => VirBinaryOp::Bitwise(BitwiseOp::BitAnd, mode),
        BinaryOp::Or => VirBinaryOp::Bitwise(BitwiseOp::BitOr, mode),
        BinaryOp::Xor => VirBinaryOp::Bitwise(BitwiseOp::BitXor, mode),
        BinaryOp::Shl => {
            VirBinaryOp::Bitwise(BitwiseOp::Shl(get_value_bitwidth(lhs, dfg), false), mode)
        }
        BinaryOp::Shr => VirBinaryOp::Bitwise(BitwiseOp::Shr(get_value_bitwidth(lhs, dfg)), mode),
    }
}

fn is_operation_between_bools(
    lhs: &ValueId,
    binary_op: &BinaryOp,
    rhs: &ValueId,
    lhs_expr: Expr,
    rhs_expr: Expr,
    dfg: &DataFlowGraph,
) -> Option<ExprX> {
    match dfg[*lhs].get_type() {
        Type::Numeric(NumericType::Unsigned { bit_size: 1 }) => {}
        _ => return None,
    }
    match dfg[*rhs].get_type() {
        Type::Numeric(NumericType::Unsigned { bit_size: 1 }) => {}
        _ => return None,
    }

    match binary_op {
        BinaryOp::And | BinaryOp::Mul => Some(ExprX::Binary(VirBinaryOp::And, lhs_expr, rhs_expr)),
        BinaryOp::Or | BinaryOp::Add => Some(ExprX::Binary(VirBinaryOp::Or, lhs_expr, rhs_expr)),
        BinaryOp::Xor => Some(ExprX::Binary(VirBinaryOp::Xor, lhs_expr, rhs_expr)),
        _ => None,
    }
}

fn array_to_expr(
    array_id: &ValueId,
    array_values: &im::Vector<ValueId>,
    noir_type: &Type,
    dfg: &DataFlowGraph,
) -> Expr {
    let vals_to_expr: Vec<Expr> =
        array_values.iter().map(|val_id| ssa_value_to_expr(val_id, dfg, None)).collect();
    SpannedTyped::new(
        &build_span(
            array_id,
            format!(
                "Array({}) with values[{}]",
                array_id.to_string(),
                array_values.iter().map(|x| x.to_string()).collect::<Vec<String>>().join(", ")
            ),
            Some(dfg.get_value_call_stack(*array_id)),
        ),
        &from_noir_type(noir_type.clone(), None),
        ExprX::ArrayLiteral(Arc::new(vals_to_expr)),
    )
}

fn param_to_expr(
    value_id: &ValueId,
    position: Option<usize>,
    noir_type: &Type,
    result_id_fixer: Option<&ResultIdFixer>,
    dfg: &DataFlowGraph,
) -> Expr {
    if let Some(result_id_fixer) = result_id_fixer {
        if let Some(expr) = result_id_fixer.fix_id(value_id) {
            return expr;
        }
    }
    let debug_string = if let Some(position_index) = position {
        "param position ".to_owned() + &position_index.to_string()
    } else {
        value_id.to_string()
    };
    SpannedTyped::new(
        &build_span(value_id, debug_string, Some(dfg.get_value_call_stack(*value_id))),
        &from_noir_type(noir_type.clone(), None),
        ExprX::Var(id_into_var_ident(value_id.clone())),
    )
}

fn numeric_const_to_expr(numeric_const: &FieldElement, noir_type: &Type) -> Expr {
    if noir_type.bit_size() == 1 {
        // Numeric const is a bool
        if numeric_const.is_zero() {
            return SpannedTyped::new(
                &empty_span(), //TODO get rid of empty span, maybe pass val_id here
                &from_noir_type(noir_type.clone(), None),
                ExprX::Const(Constant::Bool(false)),
            );
        } else {
            return SpannedTyped::new(
                &empty_span(), //TODO get rid of empty span, maybe pass val_id here
                &from_noir_type(noir_type.clone(), None),
                ExprX::Const(Constant::Bool(true)),
            );
        }
    }
    // It's an actual numeric constant
    let const_big_uint: BigUint = numeric_const.into_repr().into();
    let mut const_big_int: BigInt =
        BigInt::from_biguint(num_bigint::Sign::Plus, const_big_uint.clone());
    if let Type::Numeric(NumericType::Signed { bit_size }) = noir_type {
        if const_big_int > BigInt::from(2_u128.pow(*bit_size - 1)) {
            const_big_int -= BigInt::from(2_u128.pow(*bit_size));
        }
    }

    SpannedTyped::new(
        &empty_span(), //TODO maybe dont use empty span
        &from_noir_type(noir_type.clone(), None),
        ExprX::Const(Constant::Int(const_big_int)),
    )
}

fn ssa_value_to_expr(
    value_id: &ValueId,
    dfg: &DataFlowGraph,
    result_id_fixer: Option<&ResultIdFixer>,
) -> Expr {
    let value_id = &dfg.resolve(*value_id);
    let value = &dfg[*value_id];
    match value {
        Value::Instruction { instruction: _, position: _, typ } => {
            param_to_expr(value_id, None, typ, result_id_fixer, dfg)
        }
        Value::Param { block: _, position, typ } => {
            param_to_expr(value_id, Some(*position), typ, None, dfg)
        }
        Value::NumericConstant { constant, typ } => numeric_const_to_expr(constant, typ),
        Value::Array { array, typ } => array_to_expr(value_id, array, typ, dfg), //TODO(totel) See if there is an other way to represent arrays
        Value::Function(_) => unreachable!(), // The only possible way to have a Value::Function is through Instruction::Call
        Value::Intrinsic(_) => todo!(),       // Not planned for the prototype
        Value::ForeignFunction(_) => todo!(), // Not planned for the prototype
    }
}

fn return_values_to_expr(
    return_values_ids: &Vec<ValueId>,
    dfg: &DataFlowGraph,
    basic_block_id: Id<BasicBlock>,
) -> Option<Expr> {
    let return_values_ids: Vec<ValueId> =
        return_values_ids.iter().map(|val_id| dfg.resolve(*val_id)).collect();

    match return_values_ids.len() {
        0 => None,
        1 => {
            let mut ret_as_expr = ssa_value_to_expr(&return_values_ids[0], dfg, None);
            // Deref return value if it is a reference.
            if let TypX::Decorate(TypDecoration::Ref, _, inner_typ) = ret_as_expr.typ.as_ref() {
                ret_as_expr =
                    SpannedTyped::new(&ret_as_expr.span, inner_typ, ret_as_expr.x.clone());
            }
            Some(ret_as_expr)
        }
        _ => {
            let tuple_exprs: Exprs = Arc::new(
                return_values_ids
                    .iter()
                    .map(|val_id| ssa_value_to_expr(val_id, dfg, None))
                    .collect(),
            );
            Some(mk_tuple(
                &build_span(
                    &basic_block_id,
                    format!("Tuple of terminating instr of block({})", basic_block_id),
                    Some(dfg.get_value_call_stack(*return_values_ids.last().unwrap())),
                ),
                &tuple_exprs,
            ))
        }
    }
}

fn binary_instruction_to_expr(
    instruction_id: InstructionId,
    binary: &Binary,
    mode: Mode,
    dfg: &DataFlowGraph,
    current_context: &mut SSAContext,
) -> Expr {
    let Binary { lhs, rhs, operator } = binary;
    let lhs_expr = ssa_value_to_expr(lhs, dfg, current_context.result_id_fixer);
    let rhs_expr = ssa_value_to_expr(rhs, dfg, current_context.result_id_fixer);
    let mut binary_exprx = ExprX::Binary(
        binary_op_to_vir_binary_op(operator, mode, lhs, dfg),
        lhs_expr.clone(),
        rhs_expr.clone(),
    );

    // Special cases for operations between booleans
    if let Some(exprx) =
        is_operation_between_bools(lhs, operator, rhs, lhs_expr.clone(), rhs_expr, dfg)
    {
        binary_exprx = exprx;
        return SpannedTyped::new(
            &build_span(
                &instruction_id,
                format!("lhs({}) binary_op({}) rhs({})", lhs, operator, rhs),
                Some(dfg.get_call_stack(instruction_id)),
            ),
            &instr_res_type_to_vir_type(binary.result_type(), dfg),
            binary_exprx,
        );
    }

    let binary_expr = SpannedTyped::new(
        &build_span(
            &instruction_id,
            format!("lhs({}) binary_op({}) rhs({})", lhs, operator, rhs),
            Some(dfg.get_call_stack(instruction_id)),
        ),
        &instr_res_type_to_vir_type(binary.result_type(), dfg),
        binary_exprx,
    );

    if let Some(condition_id) = current_context.side_effects_condition {
        return wrap_with_an_if_logic(
            condition_id,
            binary_expr,
            lhs_expr,
            dfg,
            current_context.result_id_fixer,
        );
    }
    binary_expr
}
/// Depending on the bit width size we want to either return a
/// `unary boolean not` expression or a `unary bit not` expression.
fn bitwise_not_instr_to_exprx(
    value_id: &ValueId,
    dfg: &DataFlowGraph,
    bit_width: Option<IntegerTypeBitwidth>,
    result_id_fixer: Option<&ResultIdFixer>,
) -> ExprX {
    let expr = ssa_value_to_expr(value_id, dfg, result_id_fixer);

    match bit_width {
        Some(IntegerTypeBitwidth::Width(1)) => ExprX::Unary(UnaryOp::Not, expr),
        Some(width) => ExprX::Unary(UnaryOp::BitNot(Some(width)), expr),
        None => ExprX::Unary(UnaryOp::BitNot(None), expr),
    }
}

fn bitwise_not_instr_to_expr(
    value_id: &ValueId,
    dfg: &DataFlowGraph,
    result_id_fixer: Option<&ResultIdFixer>,
) -> Expr {
    let value = &dfg[*value_id];
    let bit_width: Option<IntegerTypeBitwidth> = match value.get_type() {
        Type::Numeric(numeric_type) => get_integer_bit_width(*numeric_type),
        _ => panic!("Bitwise negation on a non numeric type"),
    };
    let bitnot_exprx = bitwise_not_instr_to_exprx(value_id, dfg, bit_width, result_id_fixer);
    SpannedTyped::new(
        &build_span(
            value_id,
            format!("Unary negation on({})", value_id.to_string()),
            Some(dfg.get_value_call_stack(*value_id)),
        ),
        &from_noir_type(value.get_type().clone(), None),
        bitnot_exprx,
    )
}

fn build_const_expr(const_num: i64, value_id: &ValueId, noir_type: &Type) -> Expr {
    SpannedTyped::new(
        &build_span(value_id, format!("Const {const_num}"), None),
        &from_noir_type(noir_type.clone(), None),
        ExprX::Const(Constant::Int(BigInt::from(const_num))),
    )
}

fn usize_to_const_expr(const_num: usize, noir_type: &Type) -> Expr {
    SpannedTyped::new(
        &empty_span(),
        &from_noir_type(noir_type.clone(), None),
        ExprX::Const(Constant::Int(BigInt::from(const_num))),
    )
}

fn cast_bool_to_integer(
    value_id: &ValueId,
    noir_type: &Type,
    dfg: &DataFlowGraph,
    result_id_fixer: Option<&ResultIdFixer>,
) -> Expr {
    let if_return_type = from_noir_type(noir_type.clone(), None);
    let condition = ssa_value_to_expr(value_id, dfg, result_id_fixer);

    let const_true = build_const_expr(1, value_id, noir_type);
    let const_false = build_const_expr(0, value_id, noir_type);

    let if_true_expr = SpannedTyped::new(
        &build_span(
            value_id,
            format!("Then condition of if"),
            Some(dfg.get_value_call_stack(*value_id)),
        ),
        &if_return_type,
        ExprX::Block(Arc::new(vec![]), Some(const_true)),
    );
    let if_false_expr = SpannedTyped::new(
        &build_span(
            value_id,
            format!("Then condition of if"),
            Some(dfg.get_value_call_stack(*value_id)),
        ),
        &if_return_type,
        ExprX::Block(Arc::new(vec![]), Some(const_false)),
    );

    let if_expr = SpannedTyped::new(
        &build_span(
            value_id,
            format!("If expr because bool to int cast"),
            Some(dfg.get_value_call_stack(*value_id)),
        ),
        &if_return_type,
        ExprX::If(condition, if_true_expr, Some(if_false_expr)),
    );
    if_expr
}

fn cast_integer_to_integer(
    value_id: &ValueId,
    noir_type: &Type,
    dfg: &DataFlowGraph,
    result_id_fixer: Option<&ResultIdFixer>,
) -> Expr {
    let numeric_type = if let Type::Numeric(numeric_type) = noir_type {
        numeric_type
    } else {
        unreachable!("Can cast only to numeric types");
    };
    let cast_exprx = ExprX::Unary(
        UnaryOp::Clip { range: get_int_range(*numeric_type), truncate: false },
        ssa_value_to_expr(value_id, dfg, result_id_fixer),
    );
    SpannedTyped::new(
        &build_span(
            value_id,
            format!("Cast({}) to type({})", value_id, noir_type),
            Some(dfg.get_value_call_stack(*value_id)),
        ),
        &from_noir_type(noir_type.clone(), None),
        cast_exprx,
    )
}

fn cast_instruction_to_expr(
    value_id: &ValueId,
    noir_type: &Type,
    dfg: &DataFlowGraph,
    result_id_fixer: Option<&ResultIdFixer>,
) -> Expr {
    match dfg[*value_id].get_type() {
        Type::Numeric(NumericType::Unsigned { bit_size: 1 }) => {
            cast_bool_to_integer(value_id, noir_type, dfg, result_id_fixer)
        }
        Type::Numeric(..) => cast_integer_to_integer(value_id, noir_type, dfg, result_id_fixer),
        _ => panic!("Expected that all SSA casts have numeric targets"),
    }
}

fn range_limit_to_expr(
    value_id: &ValueId,
    target_bit_size: u32,
    truncate: bool,
    dfg: &DataFlowGraph,
    result_id_fixer: Option<&ResultIdFixer>,
) -> Expr {
    let value_type = dfg[*value_id].get_type();
    let clip_exprx = match value_type {
        Type::Numeric(numeric_type) => ExprX::Unary(
            UnaryOp::Clip {
                range: trunc_target_int_range(numeric_type, target_bit_size),
                truncate,
            },
            ssa_value_to_expr(value_id, dfg, result_id_fixer),
        ),
        _ => panic!("Can range limit/truncate only numeric values"),
    };

    let debug_string = if truncate {
        format!("Truncate var({}) to bit size({})", value_id, target_bit_size)
    } else {
        format!("Range check var({}) to bit size({})", value_id, target_bit_size)
    };
    SpannedTyped::new(
        &build_span(value_id, debug_string, Some(dfg.get_value_call_stack(*value_id))),
        &from_noir_type(value_type.clone(), None),
        clip_exprx,
    )
}

fn constrain_instruction_to_expr(
    instruction_id: Id<Instruction>,
    lhs: &ValueId,
    rhs: &ValueId,
    dfg: &DataFlowGraph,
    result_id_fixer: Option<&ResultIdFixer>,
) -> Expr {
    let binary_equals_expr = SpannedTyped::new(
        &build_span(
            &instruction_id,
            format!("lhs({}) == rhs({})", lhs, rhs),
            Some(dfg.get_call_stack(instruction_id)),
        ),
        &Arc::new(TypX::Bool),
        ExprX::Binary(
            VirBinaryOp::Eq(Mode::Spec), // Verus uses Spec for Eq expressions in asserts
            ssa_value_to_expr(lhs, dfg, result_id_fixer),
            ssa_value_to_expr(rhs, dfg, result_id_fixer),
        ),
    );
    let assert_exprx = ExprX::AssertAssume { is_assume: false, expr: binary_equals_expr };
    let assert_expr = SpannedTyped::new(
        &build_span(
            &instruction_id,
            format!("Constrain({}) lhs({}) == rhs({})", instruction_id, lhs, rhs),
            Some(dfg.get_call_stack(instruction_id)),
        ),
        &get_empty_vir_type(),
        assert_exprx,
    );
    let block_wrap = SpannedTyped::new(
        &build_span(
            &instruction_id,
            format!("Block wrapper for AssertAssume"),
            Some(dfg.get_call_stack(instruction_id)),
        ),
        &get_empty_vir_type(),
        ExprX::Block(Arc::new(vec![]), Some(assert_expr)),
    );
    SpannedTyped::new(
        &build_span(
            &instruction_id,
            format!("Ghost wrapper for AssertAssume"),
            Some(dfg.get_call_stack(instruction_id)),
        ),
        &get_empty_vir_type(),
        ExprX::Ghost { alloc_wrapper: false, tracked: false, expr: block_wrap },
    )
}

fn call_instruction_to_expr(
    call_id: InstructionId,
    value_id: &ValueId,
    arguments: &Vec<ValueId>,
    dfg: &DataFlowGraph,
    result_id_fixer: Option<&ResultIdFixer>,
) -> Expr {
    let value = &dfg[*value_id];
    let func_id = match value {
        Value::Function(func_id) => func_id,
        Value::ForeignFunction(_) => panic!("Unconstrained functions are not supported"),
        _ => unreachable!("You can only call functions in SSA"),
    };

    let name = func_id_into_funx_name(*func_id);
    let arguments_as_expr: Exprs = Arc::new(
        arguments.iter().map(|val_id| ssa_value_to_expr(val_id, dfg, result_id_fixer)).collect(),
    );
    let call_exprx: ExprX = ExprX::Call(
        CallTarget::Fun(
            CallTargetKind::Static,
            name,
            Arc::new(vec![]), // Argument types are not being passed in Rust to VIR
            Arc::new(vec![]),
            AutospecUsage::Final, // In Verus for non ghost calls they mark them as Final
        ),
        arguments_as_expr,
    );
    let function_return_type: Typ = get_function_ret_type(dfg.instruction_results(call_id), dfg);

    SpannedTyped::new(
        &build_span(
            value_id,
            format!(
                "Call({}) function({}) with args[{}]",
                value_id,
                func_id,
                arguments.iter().map(|x| x.to_string()).collect::<Vec<String>>().join(", ")
            ),
            Some(dfg.get_call_stack(call_id)),
        ),
        &function_return_type,
        call_exprx,
    )
}

fn gather_all_add_instructions(instruction_id: &InstructionId, dfg: &DataFlowGraph) -> BigInt {
    match &dfg[*instruction_id] {
        Instruction::Binary(binary) => match binary.operator {
            BinaryOp::Add => match &dfg[binary.rhs] {
                Value::NumericConstant { constant: numeric_const, typ: _ } => {
                    let const_big_uint: BigUint = numeric_const.into_repr().into();
                    let rhs_const =
                        BigInt::from_biguint(num_bigint::Sign::Plus, const_big_uint.clone());
                    // Lhs is an auto generated instruction. It's either an addition
                    // or multiplication (the multiplication for correcting the index after the flattening)
                    let lhs = if let Value::Instruction { instruction, position: _, typ: _ } =
                        dfg[binary.lhs]
                    {
                        instruction
                    } else {
                        unreachable!("Expected auto generated \"add\" instruction to have lhs of type instruction");
                    };

                    rhs_const + gather_all_add_instructions(&lhs, dfg)
                }
                _ => unreachable!("Expected auto generated \"add\" instruction to have const rhs"),
            },
            // No more auto generated add instructions
            _ => BigInt::ZERO,
        },
        // No more auto generated add instructions
        _ => BigInt::ZERO,
    }
}

#[derive(Debug)]
enum Index {
    SsaValue(ValueId),
    ConstIndex(usize),
}

impl std::fmt::Display for Index {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Index::SsaValue(id) => write!(f, "{id}"),
            Index::ConstIndex(const_index) => write!(f, "const {const_index}"),
        }
    }
}

/// In some cases we have to reverse engineer the actual value of the index.
/// This is because SSA flattens the array. Therefore an array of tuples with length n
/// we will have length of 2*n because it has been flattened.
/// For composite inner types we also have to calculate the index for the tuple.
fn calculate_index_and_tuple_index(
    index: &Index,
    inner_type_length: usize,
    dfg: &DataFlowGraph,
    result_id_fixer: Option<&ResultIdFixer>,
    mode: Mode,
) -> (Expr, Option<BigInt>) {
    // If the inner_type_length is less than 2, then the array hasn't been flattened.
    // So we just return the index transformed into a VIR expression
    if inner_type_length < 2 {
        match index {
            Index::SsaValue(id) => {
                return (ssa_value_to_expr(&id, dfg, result_id_fixer), None);
            }
            Index::ConstIndex(const_index) => {
                return (
                    usize_to_const_expr(
                        *const_index,
                        &Type::Numeric(NumericType::Unsigned { bit_size: 32 }),
                    ),
                    None,
                );
            }
        };
    }

    let index_value = match index {
        Index::SsaValue(id) => dfg[*id].clone(),
        // If we have a Index::ConstIndex we don't calculate the indexes.
        // This is needed when we want to extract the entire tuple instead of an element.
        // This happens when we are handling a SSA array set instruction.
        Index::ConstIndex(const_index) => {
            return (
                usize_to_const_expr(
                    *const_index,
                    &Type::Numeric(NumericType::Unsigned { bit_size: 32 }),
                ),
                None,
            );
        }
    };

    // There are two possible options for the index.
    // It's either a numeric constant or a Value::Instruction
    match &index_value {
        Value::Instruction { instruction, position: _, typ: noir_type } => {
            let bit_size = 32; // This is the index bit size
            let index = match index {
                Index::SsaValue(id) => id,
                Index::ConstIndex(_) => unreachable!(),
            };
            let tuple_index = gather_all_add_instructions(instruction, dfg);
            let tuple_index_as_expr = SpannedTyped::new(
                &empty_span(),
                &Arc::new(TypX::Int(IntRange::U(bit_size))),
                ExprX::Const(Constant::Int(tuple_index.clone())),
            );

            // lhs == (index - tuple_index)
            let lhs_expr = SpannedTyped::new(
                &empty_span(),
                &from_noir_type(noir_type.clone(), None),
                ExprX::Binary(
                    VirBinaryOp::Arith(ArithOp::Sub, mode),
                    ssa_value_to_expr(&index, dfg, result_id_fixer),
                    tuple_index_as_expr.clone(),
                ),
            );

            // rhs == inner_type_length
            let rhs_expr = SpannedTyped::new(
                &empty_span(),
                &Arc::new(TypX::Int(IntRange::U(bit_size))),
                ExprX::Const(Constant::Int(BigInt::from(inner_type_length))),
            );

            // actual_index = (index - tuple_index)/inner_type_length
            let actual_index_expr = SpannedTyped::new(
                &empty_span(),
                &from_noir_type(noir_type.clone(), None),
                ExprX::Binary(VirBinaryOp::Arith(ArithOp::EuclideanDiv, mode), lhs_expr, rhs_expr),
            );
            // If we are inside of an attribute we early return because we don't have to make
            // assume statments and wrap with ghost blocks. There is no overflowing in Spec code.
            match mode {
                Mode::Spec | Mode::Proof => return (actual_index_expr, Some(tuple_index)),
                Mode::Exec => {}
            };

            let comparison_expr = SpannedTyped::new(
                &empty_span(),
                &Arc::new(TypX::Bool),
                ExprX::Binary(
                    VirBinaryOp::Inequality(InequalityOp::Ge),
                    ssa_value_to_expr(&index, dfg, result_id_fixer),
                    tuple_index_as_expr,
                ),
            );

            // Assume that (index > inner_types.len())
            // This is true in SSA because SSA flattens the arrays and increases the index to match the flattened array.
            let assume_expr = SpannedTyped::new(
                &empty_span(),
                &get_empty_vir_type(),
                ExprX::AssertAssume { is_assume: true, expr: comparison_expr },
            );

            let ghost_wrap_assume_expr = SpannedTyped::new(
                &empty_span(),
                &get_empty_vir_type(),
                ExprX::Ghost { alloc_wrapper: false, tracked: false, expr: assume_expr },
            );

            let assume_stmt = Spanned::new(empty_span(), StmtX::Expr(ghost_wrap_assume_expr));
            let wrap_with_assume = SpannedTyped::new(
                &empty_span(),
                &from_noir_type(noir_type.clone(), None),
                ExprX::Block(Arc::new(vec![assume_stmt]), Some(actual_index_expr)),
            );

            (wrap_with_assume, Some(tuple_index))
        }
        Value::NumericConstant { constant: numeric_const, typ: noir_type } => {
            let const_big_uint: BigUint = numeric_const.into_repr().into();
            let const_big_int: BigInt =
                BigInt::from_biguint(num_bigint::Sign::Plus, const_big_uint.clone());

            let divisor = BigInt::from(inner_type_length);
            let tuple_index = const_big_int.clone() % divisor.clone();
            let actual_index = (const_big_int - tuple_index.clone()) / divisor;

            let actual_index_as_expr = SpannedTyped::new(
                &empty_span(),
                &from_noir_type(noir_type.clone(), None),
                ExprX::Const(Constant::Int(actual_index)),
            );

            (actual_index_as_expr, Some(tuple_index))
        }
        _ => unreachable!(
            "For a flatten array you can only index it using a constant or an instruction"
        ),
    }
}

/// Transforming an array_get instruction is tricky because we have to use a
/// function from the verus standard library. The way we do it is we generate a
/// function call to the needed vstd function. This function becomes available
/// in a later stage when we merge the noir VIR with the vstd VIR. Therefore
/// the function call becomes valid.
fn array_get_to_expr(
    array_id: &ValueId,
    index: Index,
    instruction_id: InstructionId,
    mode: Mode,
    dfg: &DataFlowGraph,
    current_context: &mut SSAContext,
) -> Expr {
    let vstd_krate = Some(Arc::new("vstd".to_string()));
    let array_return_type: Typ =
        get_function_ret_type(dfg.instruction_results(instruction_id), dfg);

    let array_length = dfg.try_get_array_length(*array_id).unwrap();
    let array_length_as_type = into_vir_const_int(array_length);
    let inner_type_noir = match dfg.type_of_value(*array_id) {
        Type::Array(inner_types, _) => inner_types,
        _ => unreachable!("You can only index an array in SSA"),
    };
    let inner_type_length = inner_type_noir.as_ref().len();
    let inner_type = from_composite_type(inner_type_noir);

    let array_inner_type_and_length_type: Typs =
        Arc::new(vec![inner_type.clone(), array_length_as_type.clone()]);

    let array_as_primary_vir_type = Arc::new(TypX::Primitive(
        Primitive::Array,
        Arc::new(vec![inner_type.clone(), array_length_as_type.clone()]),
    ));
    let array_as_vir_expr: Expr = SpannedTyped::new(
        &build_span(
            array_id,
            format!("Array{} as expr", array_id),
            Some(dfg.get_call_stack(instruction_id)),
        ),
        &Arc::new(TypX::Decorate(TypDecoration::Ref, None, array_as_primary_vir_type.clone())),
        (*ssa_value_to_expr(array_id, dfg, current_context.result_id_fixer)).x.clone(),
    );

    let index_as_vir_expr: Expr;
    let segments: Idents;
    let call_target_kind: CallTargetKind;
    let typs_for_vstd_func_call: Typs;
    let trait_impl_paths: ImplPaths;
    let autospec_usage: AutospecUsage;
    let tuple_index: Option<BigInt>;
    match mode {
        Mode::Spec => {
            segments = Arc::new(vec![
                Arc::new("array".to_string()),
                Arc::new("ArrayAdditionalSpecFns".to_string()),
                Arc::new("spec_index".to_string()),
            ]);
            let segments_for_resolved = Arc::new(vec![
                Arc::new("array".to_string()),
                Arc::new("impl&%2".to_string()),
                Arc::new("spec_index".to_string()),
            ]);
            call_target_kind = CallTargetKind::DynamicResolved {
                resolved: Arc::new(FunX {
                    path: Arc::new(PathX {
                        krate: vstd_krate.clone(),
                        segments: segments_for_resolved,
                    }),
                }),
                typs: array_inner_type_and_length_type,
                impl_paths: Arc::new(vec![]),
                is_trait_default: false,
            };
            typs_for_vstd_func_call = Arc::new(vec![array_as_primary_vir_type, inner_type.clone()]);
            let trait_impl_path1 = ImplPath::TraitImplPath(Arc::new(PathX {
                krate: vstd_krate.clone(),
                segments: Arc::new(vec![
                    Arc::new("array".to_string()),
                    Arc::new("impl&%0".to_string()),
                ]),
            }));
            let trait_impl_path2 = ImplPath::TraitImplPath(Arc::new(PathX {
                krate: vstd_krate.clone(),
                segments: Arc::new(vec![
                    Arc::new("array".to_string()),
                    Arc::new("impl&%2".to_string()),
                ]),
            }));
            trait_impl_paths = Arc::new(vec![trait_impl_path1, trait_impl_path2]);
            autospec_usage = AutospecUsage::IfMarked;
            (index_as_vir_expr, tuple_index) = calculate_index_and_tuple_index(
                &index,
                inner_type_length,
                dfg,
                current_context.result_id_fixer,
                mode,
            );
        }
        Mode::Exec => {
            segments = Arc::new(vec![
                Arc::new("array".to_string()),
                Arc::new("array_index_get".to_string()),
            ]);
            call_target_kind = CallTargetKind::Static;
            typs_for_vstd_func_call = array_inner_type_and_length_type;
            trait_impl_paths = Arc::new(vec![]);
            autospec_usage = AutospecUsage::Final;
            (index_as_vir_expr, tuple_index) = calculate_index_and_tuple_index(
                &index,
                inner_type_length,
                dfg,
                current_context.result_id_fixer,
                mode,
            );
        }
        Mode::Proof => unreachable!(), // Out of scope for the prototype
    };

    let array_get_vir_exprx: ExprX = ExprX::Call(
        CallTarget::Fun(
            call_target_kind.clone(),
            Arc::new(FunX {
                path: Arc::new(PathX { krate: vstd_krate.clone(), segments: segments.clone() }),
            }),
            typs_for_vstd_func_call.clone(),
            trait_impl_paths.clone(),
            autospec_usage,
        ),
        Arc::new(vec![array_as_vir_expr.clone(), index_as_vir_expr]),
    );
    let ref_wrapped_array_return_type: Typ =
        Arc::new(TypX::Decorate(TypDecoration::Ref, None, inner_type.clone()));

    let mut array_get_vir_expr = SpannedTyped::new(
        &build_span(
            &instruction_id,
            format!("Array get with index {}", index),
            Some(dfg.get_call_stack(instruction_id)),
        ),
        &ref_wrapped_array_return_type,
        array_get_vir_exprx,
    );

    if let Some(tuple_index) = tuple_index {
        array_get_vir_expr = SpannedTyped::new(
            &empty_span(),
            &array_return_type,
            ExprX::UnaryOpr(
                vir::ast::UnaryOpr::Field(FieldOpr {
                    datatype: Dt::Tuple(inner_type_length),
                    variant: Arc::new("tuple%".to_string() + &inner_type_length.to_string()),
                    field: Arc::new(tuple_index.to_string()),
                    get_variant: false,
                    check: vir::ast::VariantCheck::None,
                }),
                array_get_vir_expr,
            ),
        )
    }

    if current_context.quantifier_context.is_inside_quantifier_body() {
        array_get_vir_expr = SpannedTyped::new(
            &empty_span(),
            &array_return_type,
            ExprX::Unary(UnaryOp::Trigger(TriggerAnnotation::Trigger(None)), array_get_vir_expr),
        )
    }

    if let Some(condition_id) = current_context.side_effects_condition {
        let array_get_dummy = SpannedTyped::new(
            &build_span(
                &instruction_id,
                format!("Array get with index {}", index),
                Some(dfg.get_call_stack(instruction_id)),
            ),
            &ref_wrapped_array_return_type,
            ExprX::Call(
                CallTarget::Fun(
                    call_target_kind,
                    Arc::new(FunX {
                        path: Arc::new(PathX { krate: vstd_krate, segments: segments }),
                    }),
                    typs_for_vstd_func_call,
                    trait_impl_paths,
                    autospec_usage,
                ),
                Arc::new(vec![
                    array_as_vir_expr,
                    SpannedTyped::new(
                        &empty_span(),
                        &Arc::new(TypX::Int(IntRange::U(32))),
                        ExprX::Const(Constant::Int(BigInt::default())),
                    ),
                ]),
            ),
        );
        return wrap_with_an_if_logic(
            condition_id,
            array_get_vir_expr,
            array_get_dummy,
            dfg,
            current_context.result_id_fixer,
        );
    }

    array_get_vir_expr
}

/// Because array mutability is not supported in Verus we are creating a new array
/// every time there is an array mutation aka ArraySet instruction.
/// We can do it this way because the size of Noir arrays are known at compile time.
pub(crate) fn array_set_to_expr(
    array: &ValueId,
    index: &ValueId,
    new_value: &ValueId,
    instruction_id: &InstructionId,
    mode: Mode,
    current_context: &mut SSAContext,
    dfg: &DataFlowGraph,
) -> Expr {
    let array_len = dfg.try_get_array_length(*array).expect("Array id must be of type array");
    let call_stack = dfg.get_call_stack(*instruction_id);
    let mut new_array_elements: Vec<Expr> = Vec::new();

    for i in 0..array_len {
        new_array_elements.push(if_expression_for_array_body(
            &array,
            i,
            &index,
            &new_value,
            instruction_id,
            &call_stack,
            mode,
            current_context,
            dfg,
        ));
    }

    SpannedTyped::new(
        &build_span(&array, format!("array set"), Some(call_stack)),
        &from_noir_type(dfg[*array].get_type().clone(), None),
        ExprX::ArrayLiteral(Arc::new(new_array_elements)),
    )
}

fn if_expression_for_array_body(
    array_id: &ValueId,
    position_in_array: usize,
    index: &ValueId,
    new_value: &ValueId,
    instruction_id: &InstructionId,
    call_stack: &CallStack,
    mode: Mode,
    current_context: &mut SSAContext,
    dfg: &DataFlowGraph,
) -> Expr {
    let span = build_span(index, format!("If is index expression"), Some(call_stack.clone()));
    let new_value_as_expr = ssa_value_to_expr(new_value, dfg, current_context.result_id_fixer);
    let inner_type_noir = match dfg.type_of_value(*array_id) {
        Type::Array(inner_types, _) => inner_types,
        _ => unreachable!("You can only array_set an array in SSA"),
    };
    let inner_type_length = inner_type_noir.as_ref().len();
    let if_return_type = from_composite_type(inner_type_noir.clone());

    let (calculated_actual_index, tuple_index) = calculate_index_and_tuple_index(
        &Index::SsaValue(*index),
        inner_type_length,
        dfg,
        current_context.result_id_fixer,
        mode,
    );

    let if_condition = SpannedTyped::new(
        &span,
        &Arc::new(TypX::Bool),
        ExprX::Binary(
            VirBinaryOp::Eq(mode),
            calculated_actual_index,
            usize_to_const_expr(position_in_array, &dfg[*index].get_type().clone()),
        ),
    );

    // Here we will have to do tuple building with "Ctor" from 0..inner_type_length.
    // We need the calculated actual index and the special index for the ctor tuple which will be mutated.
    // Rest of the elements of the tuple will have the value old_array[actual_index].i where i in 0..inner_type_length.

    let then_expr = if let Some(tuple_index_to_be_mutated) = tuple_index {
        let mut binders: Vec<Binder<Expr>> = Vec::new();
        for current_tuple_index in 0..inner_type_length {
            let typ_for_current_index =
                from_noir_type(inner_type_noir.as_ref()[current_tuple_index].clone(), None);
            let expr_at_current_tuple_index =
                if current_tuple_index.to_bigint().expect("Failed to convert usize to BigInt")
                    == tuple_index_to_be_mutated
                {
                    new_value_as_expr.clone()
                } else {
                    let array_get_expr = array_get_to_expr(
                        array_id,
                        Index::ConstIndex(position_in_array),
                        *instruction_id,
                        mode,
                        dfg,
                        current_context,
                    );
                    SpannedTyped::new(
                        &span,
                        &typ_for_current_index,
                        ExprX::UnaryOpr(
                            vir::ast::UnaryOpr::Field(FieldOpr {
                                datatype: Dt::Tuple(inner_type_length),
                                variant: Arc::new(
                                    "tuple%".to_string() + &inner_type_length.to_string(),
                                ),
                                field: Arc::new(current_tuple_index.to_string()),
                                get_variant: false,
                                check: vir::ast::VariantCheck::None,
                            }),
                            array_get_expr,
                        ),
                    )
                };
            binders.push(Arc::new(BinderX {
                name: Arc::new(current_tuple_index.to_string()),
                a: expr_at_current_tuple_index,
            }));
        }
        let build_tuple_exprx = ExprX::Ctor(
            Dt::Tuple(inner_type_length),
            Arc::new(format!("tuple%{inner_type_length}")),
            Arc::new(binders),
            None,
        );
        SpannedTyped::new(&span, &if_return_type, build_tuple_exprx)
    } else {
        SpannedTyped::new(
            &span,
            &if_return_type,
            ExprX::Block(Arc::new(vec![]), Some(new_value_as_expr)),
        )
    };

    // This array get call returns a literal array get for the given array
    // where no calculations have been done for the index.
    let else_expr = array_get_to_expr(
        array_id,
        Index::ConstIndex(position_in_array),
        *instruction_id,
        mode,
        dfg,
        current_context,
    );

    SpannedTyped::new(&span, &if_return_type, ExprX::If(if_condition, then_expr, Some(else_expr)))
}

fn quantifier_to_expr(
    instruction_id: &InstructionId,
    quant_type: QuantifierType,
    return_val: ValueId,
    dfg: &DataFlowGraph,
    current_context: &mut SSAContext,
) -> Expr {
    let (quant_indexes_opt, quant_body_opt) =
        current_context.quantifier_context.finish_quantifier();

    let quant_indexes =
        quant_indexes_opt.expect("A quantifier start instruction should have been created");
    let quant_body =
        quant_body_opt.expect("A quantifier start instruction should have been created");

    let quant_vir_type = match quant_type {
        QuantifierType::Forall => Quant { quant: AirQuant::Forall },
        QuantifierType::Exists => Quant { quant: AirQuant::Exists },
    };
    let quant_vir_indexes: Vec<VarBinder<Typ>> = quant_indexes
        .into_iter()
        .map(|index| {
            let index_val_id = ValueId::new(
                extract_quant_index_id(&index).expect("All indexes should be value ids to string"),
            );
            let index_type = dfg.type_of_value(index_val_id);
            VarBinderX {
                name: VarIdent(
                    Arc::new(index.clone()),
                    vir::ast::VarIdentDisambiguate::RustcId(index_val_id.to_usize()),
                ),
                a: if let Type::Numeric(NumericType::Unsigned { .. }) = index_type {
                    Arc::new(TypX::Int(IntRange::Nat))
                } else {
                    Arc::new(TypX::Int(IntRange::Int))
                },
            }
        })
        .map(|var_binder| Arc::new(var_binder))
        .collect();

    let quantifier_vir_body = SpannedTyped::new(
        &build_span(
            instruction_id,
            format!("{} body expression", quant_type),
            Some(dfg.get_call_stack(*instruction_id)),
        ),
        &Arc::new(TypX::Bool), // All quantifier bodies must be of type bool.
        ExprX::Block(
            Arc::new(quant_body),
            Some(ssa_value_to_expr(&return_val, dfg, current_context.result_id_fixer)),
        ),
    );
    let quantifier_vir_exprx =
        ExprX::Quant(quant_vir_type, Arc::new(quant_vir_indexes), quantifier_vir_body);

    SpannedTyped::new(
        &build_span(
            instruction_id,
            format!("{} expression", quant_type),
            Some(dfg.get_call_stack(*instruction_id)),
        ),
        &Arc::new(TypX::Bool),
        quantifier_vir_exprx,
    )
}

fn extract_quant_index_id(quant_index: &str) -> Option<usize> {
    let mut chars = quant_index.chars();
    if let Some(first_char) = chars.next() {
        if first_char == 'v' {
            let val_id: String = chars.collect();
            return val_id.parse::<usize>().ok();
        }
    } 
    None
}

fn store_to_expr(
    address: &ValueId,
    value: &ValueId,
    dfg: &DataFlowGraph,
    current_context: &mut SSAContext,
    instruction_id: &InstructionId,
) -> Expr {
    let address_type = if let Type::Reference(inner_type) = dfg.type_of_value(*address) {
        inner_type.as_ref().clone()
    } else {
        dfg.type_of_value(*address)
    };
    let lhs = SpannedTyped::new(
        &build_span(address, format!("Lhs of assign"), Some(dfg.get_call_stack(*instruction_id))),
        &from_noir_type(address_type, None),
        ExprX::VarLoc(id_into_var_ident(*address)),
    );

    SpannedTyped::new(
        &build_span(address, format!("Assign expr"), Some(dfg.get_call_stack(*instruction_id))),
        &get_empty_vir_type(),
        ExprX::Assign {
            init_not_mut: true,
            // lhs: ssa_value_to_expr(address, dfg, current_context.result_id_fixer),
            lhs,
            rhs: ssa_value_to_expr(value, dfg, current_context.result_id_fixer),
            op: None,
        },
    )
}

pub(crate) fn instruction_to_expr(
    instruction_id: InstructionId,
    instruction: &Instruction,
    mode: Mode,
    dfg: &DataFlowGraph,
    current_context: &mut SSAContext,
) -> Expr {
    match instruction {
        Instruction::Binary(binary) => {
            binary_instruction_to_expr(instruction_id, binary, mode, dfg, current_context)
        }
        Instruction::Cast(val_id, noir_type) => {
            cast_instruction_to_expr(val_id, noir_type, dfg, current_context.result_id_fixer)
        }
        Instruction::Not(val_id) => {
            bitwise_not_instr_to_expr(val_id, dfg, current_context.result_id_fixer)
        }
        Instruction::Truncate { value: val_id, bit_size, max_bit_size: _ } => {
            range_limit_to_expr(val_id, *bit_size, true, dfg, current_context.result_id_fixer)
        }
        Instruction::Constrain(lhs, rhs, _) => constrain_instruction_to_expr(
            instruction_id,
            lhs,
            rhs,
            dfg,
            current_context.result_id_fixer,
        ),
        Instruction::RangeCheck { value: val_id, max_bit_size, assert_message: _ } => {
            range_limit_to_expr(val_id, *max_bit_size, false, dfg, current_context.result_id_fixer)
        }
        Instruction::Call { func, arguments } => call_instruction_to_expr(
            instruction_id,
            func,
            arguments,
            dfg,
            current_context.result_id_fixer,
        ),
        Instruction::Allocate => unreachable!(), // Should return empty expression, that's why it is skipped
        Instruction::Load { address: value_id } => {
            ssa_value_to_expr(value_id, dfg, current_context.result_id_fixer)
        }
        Instruction::Store { address, value } => {
            store_to_expr(address, value, dfg, current_context, &instruction_id)
        }
        Instruction::EnableSideEffectsIf { condition: _ } => todo!(), //TODO(totel) Support for mutability
        Instruction::ArrayGet { array, index } => array_get_to_expr(
            array,
            Index::SsaValue(index.clone()),
            instruction_id,
            mode,
            dfg,
            current_context,
        ),
        Instruction::ArraySet { array, index, value: new_value, mutable: _ } => {
            array_set_to_expr(array, index, new_value, &instruction_id, mode, current_context, dfg)
        }
        Instruction::IncrementRc { value: _ } => unreachable!(), // Only in Brillig
        Instruction::DecrementRc { value: _ } => unreachable!(), // Only in Brillig
        Instruction::IfElse {
            then_condition: _,
            then_value: _,
            else_condition: _,
            else_value: _,
        } => todo!(),
        Instruction::QuantStart { .. } => {
            unreachable!("We skip those instructions but we mark their presence in the structure current context")
        }
        Instruction::QuantEnd { quant_type, body_expr } => {
            quantifier_to_expr(&instruction_id, *quant_type, *body_expr, dfg, current_context)
        }
    }
}

fn terminating_instruction_to_expr(
    basic_block_id: Id<BasicBlock>, // The id of the block where the terminating instr is located
    terminating_instruction: &TerminatorInstruction,
    dfg: &DataFlowGraph,
) -> Expr {
    match terminating_instruction {
        TerminatorInstruction::Return { return_values, call_stack } => {
            let return_type = get_function_ret_type(&return_values, dfg);
            let return_exprx =
                ExprX::Return(return_values_to_expr(return_values, dfg, basic_block_id));
            SpannedTyped::new(
                &build_span(
                    &basic_block_id,
                    format!("Terminating instruction of block({}) return vals", basic_block_id),
                    Some(call_stack.clone()),
                ),
                &return_type,
                return_exprx,
            )
        }
        _ => unreachable!(), // See why Jmp and JmpIf are unreachable here https://coda.io/d/_d6vM0kjfQP6#Blocksense-Table-View_tuvTVcZS/r1381&view=center
    }
}

pub(crate) fn is_instruction_enable_side_effects(
    instruction_id: &InstructionId,
    dfg: &DataFlowGraph,
) -> bool {
    match dfg[*instruction_id] {
        Instruction::EnableSideEffectsIf { condition: _ } => true,
        _ => false,
    }
}

pub(crate) fn get_enable_side_effects_value_id(
    instruction_id: &InstructionId,
    dfg: &DataFlowGraph,
) -> Option<ValueId> {
    match dfg[*instruction_id] {
        Instruction::EnableSideEffectsIf { condition } => match &dfg[condition] {
            Value::NumericConstant { constant: _, typ: _ } => None,
            Value::Instruction { instruction: _, position: _, typ: _ } => Some(condition),
            Value::Param { block: _, position: _, typ: _ } => Some(condition),
            _ => unreachable!(),
        },
        _ => unreachable!(),
    }
}

pub(crate) fn is_instruction_call_to_print(
    instruction_id: &InstructionId,
    dfg: &DataFlowGraph,
) -> bool {
    match &dfg[*instruction_id] {
        Instruction::Call { func, arguments: _ } => {
            if let Value::ForeignFunction(func_name) = &dfg[*func] {
                return func_name == "print";
            } else {
                false
            }
        }
        _ => false,
    }
}

pub(crate) fn is_instruction_ind_dec_rc(
    instruction_id: &InstructionId,
    dfg: &DataFlowGraph,
) -> bool {
    match &dfg[*instruction_id] {
        Instruction::DecrementRc { .. } | Instruction::IncrementRc { .. } => true,
        _ => false,
    }
}

/// Returns a SSA block as an expression and
/// the type of the SSA block's terminating instruction
pub(crate) fn basic_block_to_exprx(
    basic_block_id: Id<BasicBlock>,
    dfg: &DataFlowGraph,
    current_context: &mut SSAContext,
) -> (ExprX, Typ) {
    let basic_block = dfg[basic_block_id].clone();
    let mut vir_statements: Vec<Stmt> = Vec::new();
    current_context.result_id_fixer = None;

    for instruction_id in basic_block.instructions() {
        if is_instruction_enable_side_effects(instruction_id, dfg) {
            current_context.side_effects_condition =
                get_enable_side_effects_value_id(instruction_id, dfg);
        }

        if !is_instruction_enable_side_effects(instruction_id, dfg)
            && !is_instruction_call_to_print(instruction_id, dfg)
            && !is_instruction_ind_dec_rc(instruction_id, dfg)
        {
            let statement = instruction_to_stmt(
                &dfg[*instruction_id],
                dfg,
                *instruction_id,
                Mode::Exec,
                current_context,
            );
            vir_statements.push(statement);
        }
    }

    assert!(
        basic_block.terminator().is_some(),
        "All finished SSA blocks must have a terminating instruction"
    );

    let terminating_instruction = basic_block.terminator().unwrap();
    let block_ending_expr =
        terminating_instruction_to_expr(basic_block_id, terminating_instruction, dfg);
    (
        ExprX::Block(Arc::new(vir_statements), Some(block_ending_expr.clone())),
        block_ending_expr.typ.clone(),
    )
}
