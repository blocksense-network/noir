use expr_to_vir::{
    patterns::instruction_to_stmt,
    types::{
        from_noir_type, get_empty_vir_type, get_function_ret_type, get_int_range,
        get_integer_bit_width, instr_res_type_to_vir_type, into_vir_const_int,
        trunc_target_int_range,
    },
};

use crate::ssa::verus_vir_gen::*;

fn get_value_bitwidth(value_id: &ValueId, dfg: &DataFlowGraph) -> IntegerTypeBitwidth {
    let value = &dfg[*value_id];
    match value.get_type() {
        Type::Numeric(numeric_type) => get_integer_bit_width(*numeric_type).unwrap(),
        _ => panic!("Bitwise operation on a non numeric type"),
    }
}

fn wrap_with_an_if_logic(condition_id: ValueId, binary_expr: Expr, lhs_expr: Expr, dfg: &DataFlowGraph, result_id_fixer: Option<&ResultIdFixer>) -> Expr {
    let lhs_type = lhs_expr.typ.clone();
    let if_exprx = ExprX::If(ssa_value_to_expr(&condition_id, dfg, result_id_fixer), binary_expr, Some(lhs_expr));
    SpannedTyped::new(&build_span(&condition_id, format!("Enable side effects if")), &lhs_type, if_exprx)
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
        ),
        &from_noir_type(noir_type.clone(), None),
        ExprX::ArrayLiteral(Arc::new(vals_to_expr)),
    )
}

fn param_to_expr(
    value_id: &ValueId,
    position: usize,
    noir_type: &Type,
    result_id_fixer: Option<&ResultIdFixer>,
) -> Expr {
    if let Some(result_id_fixer) = result_id_fixer {
        if let Some(expr) = result_id_fixer.fix_id(value_id) {
            return expr;
        }
    }
    SpannedTyped::new(
        &build_span(value_id, "param position ".to_owned() + &position.to_string()),
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
        Value::Instruction { instruction: _, position, typ } => {
            param_to_expr(value_id, *position, typ, result_id_fixer)
        }
        Value::Param { block: _, position, typ } => param_to_expr(value_id, *position, typ, None),
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
        1 => Some(ssa_value_to_expr(&return_values_ids[0], dfg, None)),
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
    if let Some(exprx) = is_operation_between_bools(lhs, operator, rhs, lhs_expr.clone(), rhs_expr, dfg) {
        binary_exprx = exprx;
        return SpannedTyped::new(
            &build_span(&instruction_id, format!("lhs({}) binary_op({}) rhs({})", lhs, operator, rhs)),
            &instr_res_type_to_vir_type(binary.result_type(), dfg),
            binary_exprx,
        );
    }

    let binary_expr = SpannedTyped::new(
        &build_span(&instruction_id, format!("lhs({}) binary_op({}) rhs({})", lhs, operator, rhs)),
        &instr_res_type_to_vir_type(binary.result_type(), dfg),
        binary_exprx,
    );

    if let Some(condition_id) = current_context.side_effects_condition {
        return wrap_with_an_if_logic(condition_id, binary_expr, lhs_expr, dfg, current_context.result_id_fixer)
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
        &build_span(value_id, format!("Unary negation on({})", value_id.to_string())),
        &from_noir_type(value.get_type().clone(), None),
        bitnot_exprx,
    )
}

fn build_const_expr(const_num: i64, value_id: &ValueId, noir_type: &Type) -> Expr {
    SpannedTyped::new(
        &build_span(value_id, format!("Const {const_num}")),
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
        &build_span(value_id, format!("Then condition of if")),
        &if_return_type,
        ExprX::Block(Arc::new(vec![]), Some(const_true)),
    );
    let if_false_expr = SpannedTyped::new(
        &build_span(value_id, format!("Then condition of if")),
        &if_return_type,
        ExprX::Block(Arc::new(vec![]), Some(const_false)),
    );

    let if_expr = SpannedTyped::new(
        &build_span(value_id, format!("If expr because bool to int cast")),
        &if_return_type,
        ExprX::If(condition, if_true_expr, Some(if_false_expr)),
    );
    if_expr
}

fn cast_integer_to_integer(
    value_id: &ValueId,
    noir_type: &Type,
    dfg: &DataFlowGraph,
    numeric_type: &NumericType,
    result_id_fixer: Option<&ResultIdFixer>,
) -> Expr {
    let cast_exprx = ExprX::Unary(
        UnaryOp::Clip { range: get_int_range(*numeric_type), truncate: false },
        ssa_value_to_expr(value_id, dfg, result_id_fixer),
    );
    SpannedTyped::new(
        &build_span(value_id, format!("Cast({}) to type({})", value_id, noir_type)),
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
        Type::Numeric(numeric_type) => {
            cast_integer_to_integer(value_id, noir_type, dfg, numeric_type, result_id_fixer)
        }
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
        &build_span(value_id, debug_string),
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
        &build_span(&instruction_id, format!("lhs({}) == rhs({})", lhs, rhs)),
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
        ),
        &get_empty_vir_type(),
        assert_exprx,
    );
    let block_wrap = SpannedTyped::new(
        &build_span(&instruction_id, format!("Block wrapper for AssertAssume")),
        &get_empty_vir_type(),
        ExprX::Block(Arc::new(vec![]), Some(assert_expr)),
    );
    SpannedTyped::new(
        &build_span(&instruction_id, format!("Ghost wrapper for AssertAssume")),
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
        ),
        &function_return_type,
        call_exprx,
    )
}

/// Transforming an array_get instruction is tricky because we have to use a
/// function from the verus standard library. The way we do it is we generate a
/// function call to the needed vstd function. This function becomes available
/// in a later stage when we merge the noir VIR with the vstd VIR. Therefore
/// the function call becomes valid.
fn array_get_to_expr(
    array_id: &ValueId,
    index: &ValueId,
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
    let array_inner_type_and_length_type: Typs =
        Arc::new(vec![array_return_type.clone(), array_length_as_type.clone()]);
    let array_as_primary_vir_type = Arc::new(TypX::Primitive(
        Primitive::Array,
        Arc::new(vec![array_return_type.clone(), array_length_as_type.clone()]),
    ));
    let array_as_vir_expr: Expr = SpannedTyped::new(
        &build_span(array_id, format!("Array{} as expr", array_id)),
        &Arc::new(TypX::Decorate(TypDecoration::Ref, None, array_as_primary_vir_type.clone())),
        (*ssa_value_to_expr(array_id, dfg, current_context.result_id_fixer)).x.clone(),
    );
    let index_as_vir_expr: Expr;
    let segments: Idents;
    let call_target_kind: CallTargetKind;
    let typs_for_vstd_func_call: Typs;
    let trait_impl_paths: ImplPaths;
    let autospec_usage: AutospecUsage;
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
            typs_for_vstd_func_call =
                Arc::new(vec![array_as_primary_vir_type, array_return_type.clone()]);
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
            index_as_vir_expr = ssa_value_to_expr(index, dfg, current_context.result_id_fixer);
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
            index_as_vir_expr = ssa_value_to_expr(index, dfg, current_context.result_id_fixer);
        }
        Mode::Proof => unreachable!(), // Out of scope for the prototype
    };

    let array_get_vir_exprx: ExprX = ExprX::Call(
        CallTarget::Fun(
            call_target_kind.clone(),
            Arc::new(FunX { path: Arc::new(PathX { krate: vstd_krate.clone(), segments: segments.clone() }) }),
            typs_for_vstd_func_call.clone(),
            trait_impl_paths.clone(),
            autospec_usage,
        ),
        Arc::new(vec![array_as_vir_expr.clone(), index_as_vir_expr]),
    );
    let ref_wrapped_array_return_type: Typ =
        Arc::new(TypX::Decorate(TypDecoration::Ref, None, array_return_type));

    let array_get_vir_expr = SpannedTyped::new(
        &build_span(&instruction_id, format!("Array get with index {}", index)),
        &ref_wrapped_array_return_type,
        array_get_vir_exprx,
    );

    if let Some(condition_id) = current_context.side_effects_condition {
        let array_get_dummy = SpannedTyped::new(
            &build_span(&instruction_id, format!("Array get with index {}", index)),
            &ref_wrapped_array_return_type,
            ExprX::Call(
                CallTarget::Fun(
                    call_target_kind,
                    Arc::new(FunX { path: Arc::new(PathX { krate: vstd_krate, segments: segments }) }),
                    typs_for_vstd_func_call,
                    trait_impl_paths,
                    autospec_usage,
                ),
                Arc::new(vec![array_as_vir_expr, SpannedTyped::new(
                    &empty_span(),
                    &Arc::new(TypX::Int(IntRange::U(32))),
                    ExprX::Const(Constant::Int(BigInt::default())),
                )]),
            ),
        );
        return wrap_with_an_if_logic(condition_id, array_get_vir_expr, array_get_dummy, dfg, current_context.result_id_fixer)
    }

    array_get_vir_expr
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
        Instruction::Allocate => unreachable!(), // Optimized away
        Instruction::Load { address: _ } => unreachable!(), // Optimized away
        Instruction::Store { address: _, value: _ } => unreachable!(), // Optimized away
        Instruction::EnableSideEffectsIf { condition: _ } => todo!(), //TODO(totel) Support for mutability
        Instruction::ArrayGet { array, index } => array_get_to_expr(
            array,
            index,
            instruction_id,
            mode,
            dfg,
            current_context,
        ),
        Instruction::ArraySet { array: _, index: _, value: _, mutable: _ } => {
            todo!("Array set not implemented")
        }
        Instruction::IncrementRc { value: _ } => unreachable!(), // Only in Brillig
        Instruction::DecrementRc { value: _ } => unreachable!(), // Only in Brillig
        Instruction::IfElse {
            then_condition: _,
            then_value: _,
            else_condition: _,
            else_value: _,
        } => todo!(),
    }
}

fn terminating_instruction_to_expr(
    basic_block_id: Id<BasicBlock>, // The id of the block where the terminating instr is located
    terminating_instruction: &TerminatorInstruction,
    dfg: &DataFlowGraph,
) -> Expr {
    match terminating_instruction {
        TerminatorInstruction::Return { return_values, call_stack: _ } => {
            let return_type = get_function_ret_type(&return_values, dfg);
            let return_exprx =
                ExprX::Return(return_values_to_expr(return_values, dfg, basic_block_id));
            SpannedTyped::new(
                &build_span(
                    &basic_block_id,
                    format!("Terminating instruction of block({}) return vals", basic_block_id),
                ),
                &return_type,
                return_exprx,
            )
        }
        _ => unreachable!(), // See why Jmp and JmpIf are unreachable here https://coda.io/d/_d6vM0kjfQP6#Blocksense-Table-View_tuvTVcZS/r1381&view=center
    }
}

fn is_instruction_enable_side_effects(instruction_id: &InstructionId, dfg: &DataFlowGraph) -> bool {
    match dfg[*instruction_id] {
        Instruction::EnableSideEffectsIf { condition: _ } => true,
        _ => false,
    }
}

fn get_enable_side_effects_value_id(
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

fn is_instruction_call_to_print(instruction_id: &InstructionId, dfg: &DataFlowGraph) -> bool {
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