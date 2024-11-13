use std::sync::Arc;

use acvm::{AcirField, FieldElement};
use num_bigint::{BigInt, BigUint};
use vir::{
    ast::{
        ArithOp, AutospecUsage, Binders, BitwiseOp, CallTarget, CallTargetKind, Constant, Dt, Expr,
        ExprX, Exprs, Fun, FunX, FunctionAttrs, FunctionAttrsX, FunctionKind, FunctionX,
        GenericBounds, Idents, InequalityOp, IntRange, IntegerTypeBitwidth, ItemKind, Krate,
        KrateX, Mode, Module, ModuleX, Param, ParamX, Params, PathX, Pattern, PatternX, Primitive,
        SpannedTyped, Stmt, StmtX, Typ, TypDecoration, TypX, Typs, UnaryOp, VarIdent, Visibility,
    },
    ast_util::mk_tuple,
    def::{prefix_tuple_variant, Spanned},
    messages::Span,
};

use vir::ast::BinaryOp as VirBinaryOp;

use super::{
    ir::{
        basic_block::BasicBlock,
        dfg::DataFlowGraph,
        function::{Function, FunctionId},
        instruction::{
            Binary, BinaryOp, FvInstruction, Instruction, InstructionId, InstructionResultType,
            TerminatorInstruction,
        },
        map::Id,
        types::{CompositeType, NumericType, Type},
        value::{Value, ValueId},
    },
    ssa_gen::Ssa,
};

/// Should be named differently
#[derive(Debug)]
pub enum BuildingKrateError {
    SomeError(String),
}

fn func_id_into_segments(function_id: FunctionId) -> Idents {
    Arc::new(vec![Arc::new(function_id.to_string())]) // If I use function_id.to_string() it will return "f{id}" instead of only id
}

fn func_id_into_funx_name(function_id: FunctionId) -> Fun {
    Arc::new(FunX {
        path: Arc::new(PathX { krate: None, segments: func_id_into_segments(function_id) }),
    })
}

/// It seems that SSA format has simplified the code a lot, so I assume that every function in SSA is of type Static.
fn get_func_kind(_func: &Function) -> FunctionKind {
    FunctionKind::Static
}
/// In Verus VIR to represent a "no type" you have to return an empty tuple
fn get_empty_vir_type() -> Typ {
    Arc::new(TypX::Datatype(Dt::Tuple(0), Arc::new(Vec::new()), Arc::new(Vec::new())))
}

/// This function is technical debt. Its use should be minimized
fn empty_span() -> Span {
    Span { raw_span: Arc::new(()), id: 0, data: vec![], as_string: String::new() }
}

fn build_span<A>(ast_id: &Id<A>, debug_string: String) -> Span {
    Span {
        raw_span: Arc::new(()),       // No idea
        id: ast_id.to_usize() as u64, // AST id
        data: Vec::new(),             // No idea
        as_string: debug_string, // It's used as backup if there is no other way to show where the error comes from.
    }
}

fn empty_vec_idents() -> Idents {
    Arc::new(vec![])
}

fn empty_vec_generic_bounds() -> GenericBounds {
    Arc::new(vec![])
}

fn get_integer_bit_width(numeric_type: NumericType) -> Option<IntegerTypeBitwidth> {
    match numeric_type {
        NumericType::Signed { bit_size: _ } => None, // Expected behavior in Verus VIR
        NumericType::Unsigned { bit_size } => Some(IntegerTypeBitwidth::Width(bit_size)),
        NumericType::NativeField => Some(IntegerTypeBitwidth::Width(FieldElement::max_num_bits())),
    }
}

fn get_int_range(numeric_type: NumericType) -> IntRange {
    match numeric_type {
        NumericType::Signed { bit_size } => IntRange::I(bit_size),
        NumericType::Unsigned { bit_size } => IntRange::U(bit_size),
        NumericType::NativeField => IntRange::U(FieldElement::max_num_bits()), // TODO(totel) Document mapping Noir Fields
    }
}

fn trunc_target_int_range(numeric_type: &NumericType, target_bit_size: u32) -> IntRange {
    match numeric_type {
        NumericType::Signed { bit_size: _ } => IntRange::I(target_bit_size),
        NumericType::Unsigned { bit_size: _ } => IntRange::U(target_bit_size),
        NumericType::NativeField => IntRange::U(target_bit_size),
    }
}

fn from_numeric_type(numeric_type: NumericType) -> Typ {
    match numeric_type {
        NumericType::Signed { bit_size } => Arc::new(TypX::Int(IntRange::I(bit_size))),
        NumericType::Unsigned { bit_size } => {
            if bit_size == 1 {
                Arc::new(TypX::Bool)
            } else {
                Arc::new(TypX::Int(IntRange::U(bit_size)))
            }
        }
        NumericType::NativeField => Arc::new(TypX::Int(IntRange::U(FieldElement::max_num_bits()))), // TODO(totel) Document mapping Noir Fields
    }
}

fn into_vir_const_int(number: usize) -> Typ {
    Arc::new(TypX::ConstInt(BigInt::from(number)))
}

fn is_function_type(val: &Value) -> bool {
    match val {
        Value::Function(_) | Value::Intrinsic(_) | Value::ForeignFunction(_) => true,
        _ => false,
    }
}

fn get_func_id(val: &Value) -> FunctionId {
    match val {
        Value::Function(func_id) => func_id.clone(),
        _ => unreachable!(),
    }
}

/// Maps the Noir type of the result of an instruction to Verus VIR type
fn instr_res_type_to_vir_type(instr_res_type: InstructionResultType, dfg: &DataFlowGraph) -> Typ {
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
fn from_composite_type(composite_type: Arc<CompositeType>) -> Typ {
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
fn from_noir_type(noir_typ: Type, func_id: Option<FunctionId>) -> Typ {
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

fn build_tuple_type(values: &Vec<ValueId>, dfg: &DataFlowGraph) -> Typ {
    let datatype: Dt = Dt::Tuple(values.len());
    let tuple_types: Typs = Arc::new(
        values.iter().map(|val_id| from_noir_type(dfg[*val_id].get_type().clone(), None)).collect(),
    );
    Arc::new(TypX::Datatype(datatype, tuple_types, Arc::new(vec![])))
}

fn get_function_ret_type(results: &[Id<Value>], dfg: &DataFlowGraph) -> Typ {
    match results.len() {
        0 => get_empty_vir_type(),
        1 => from_noir_type(dfg[results[0]].get_type().clone(), None),
        _ => build_tuple_type(&results.to_vec(), dfg),
    }
}

fn id_into_var_ident(value_id: ValueId) -> VarIdent {
    VarIdent(
        Arc::new(value_id.to_string()),
        vir::ast::VarIdentDisambiguate::RustcId(value_id.to_usize()),
    )
}

/// I believe that in Verus VIR this the way they represent return var identifiers
fn return_var_ident() -> VarIdent {
    VarIdent(Arc::new(vir::def::RETURN_VALUE.to_owned()), vir::ast::VarIdentDisambiguate::AirLocal)
}
/// Probably need to be swapped with the return_var_ident function
fn empty_var_ident() -> VarIdent {
    VarIdent(Arc::new("empty_tuple".to_string()), vir::ast::VarIdentDisambiguate::NoBodyParam)
}

/// Returns a Verus VIR param
fn build_param(
    value_id: ValueId,
    vir_type: Typ,
    mode: Mode,
    is_mut: bool,
    unwrapped_info: Option<(Mode, VarIdent)>,
    position: Option<usize>, // In some cases there is no way to indicate a position
) -> Param {
    let paramx = ParamX {
        name: id_into_var_ident(value_id),
        typ: vir_type,
        mode: mode,                     // For now all parameters are of type Exec
        is_mut: is_mut,                 // As far as I understand there is no &mut in SSA
        unwrapped_info: unwrapped_info, // Only if the parameter uses Ghost(x)/Tracked(x) pattern
    };
    Spanned::new(
        build_span(&value_id, "param position ".to_owned() + &position.unwrap_or(0).to_string()),
        paramx,
    )
}

fn build_tuple_return_param(values: &Vec<ValueId>, basic_block_id: Id<BasicBlock>, dfg: &DataFlowGraph) -> Param {
    let paramx = ParamX {
        name: VarIdent(Arc::new("result".to_string()), vir::ast::VarIdentDisambiguate::NoBodyParam),
        typ: get_function_ret_type(values, dfg),
        mode: Mode::Exec,
        is_mut: false,
        unwrapped_info: None,
    };
    Spanned::new(
        build_span(&basic_block_id, "Tuple param".to_string()),
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
            return Ok(build_param(value_id, vir_type, Mode::Exec, false, None, Some(position)));
        }
        _ => {
            return Err(BuildingKrateError::SomeError(
                "expected SSA param value, found something else".to_string(),
            ))
        }
    };
}

fn get_function_params(func: &Function) -> Result<Params, BuildingKrateError> {
    let entry_block_id = func.entry_block();
    let entry_block = func.dfg[entry_block_id].clone();

    let mut parameters: Vec<Param> = Vec::new();
    for value_id in entry_block.parameters().iter() {
        let param = ssa_param_into_vir_param(*value_id, &func.dfg)?;
        parameters.push(param);
    }
    Ok(Arc::new(parameters))
}

fn is_function_return_void(func: &Function) -> bool {
    !func.returns().is_empty()
}

fn get_function_return_values(func: &Function) -> Result<Vec<ValueId>, BuildingKrateError> {
    let entry_block_id = func.entry_block();
    let terminating_instruction = func.dfg[entry_block_id].terminator();

    match terminating_instruction {
        Some(instruction) => match instruction {
            TerminatorInstruction::Return { return_values, call_stack: _ } => {
                let return_values: Vec<ValueId> =
                    return_values.iter().map(|val_id| func.dfg.resolve(*val_id)).collect();
                Ok(return_values)
            }
            _ => unreachable!(), // Only Brillig functions have a non Return Terminating instruction
        },
        None => {
            return Err(BuildingKrateError::SomeError(
                "Found None as a terminating instruction in a finished SSA block".to_string(),
            ))
        }
    }
}

fn get_function_return_param(func: &Function) -> Result<Param, BuildingKrateError> {
    let return_values = get_function_return_values(func)?;
    Ok(build_tuple_return_param(&return_values, func.entry_block(), &func.dfg))
}

/// Returns default instance of FunctionAttrs
/// By default we mean the same way a default instance would be
/// constructed in Verus VIR
fn build_default_funx_attrs(zero_args: bool) -> FunctionAttrs {
    Arc::new(FunctionAttrsX {
        uses_ghost_blocks: true,
        inline: false,
        hidden: Arc::new(vec![]), // Default in Verus
        broadcast_forall: false,
        broadcast_forall_only: false,
        no_auto_trigger: false,
        custom_req_err: None, // Can actually be used to report errors
        autospec: None,
        bit_vector: false, // Verify using bit vector theory?
        atomic: false,     // Maybe only ghost functions are atomic
        integer_ring: false,
        is_decrease_by: false,
        check_recommends: false,
        nonlinear: false,
        spinoff_prover: false,
        memoize: false,
        rlimit: None,
        print_zero_args: zero_args, // Has no default value
        print_as_method: false,
        prophecy_dependent: false,
        size_of_broadcast_proof: false,
        is_type_invariant_fn: false,
    })
}

// TODO MOVE into a separate file all expr mapping logic
fn is_multiplication_between_bools(
    lhs: &ValueId,
    rhs: &ValueId,
    binary_op: &BinaryOp,
    dfg: &DataFlowGraph,
) -> bool {
    match (dfg[*lhs].get_type(), dfg[*rhs].get_type(), binary_op) {
        (
            Type::Numeric(NumericType::Unsigned { bit_size: 1 }),
            Type::Numeric(NumericType::Unsigned { bit_size: 1 }),
            BinaryOp::Mul,
        ) => true,
        (_, _, _) => false,
    }
}

fn is_or_between_bools(
    lhs: &ValueId,
    rhs: &ValueId,
    binary_op: &BinaryOp,
    dfg: &DataFlowGraph,
) -> bool {
    match (dfg[*lhs].get_type(), dfg[*rhs].get_type(), binary_op) {
        (
            Type::Numeric(NumericType::Unsigned { bit_size: 1 }),
            Type::Numeric(NumericType::Unsigned { bit_size: 1 }),
            BinaryOp::Or,
        ) => true,
        (_, _, _) => false,
    }
}

fn is_and_between_bools(
    lhs: &ValueId,
    rhs: &ValueId,
    binary_op: &BinaryOp,
    dfg: &DataFlowGraph,
) -> bool {
    match (dfg[*lhs].get_type(), dfg[*rhs].get_type(), binary_op) {
        (
            Type::Numeric(NumericType::Unsigned { bit_size: 1 }),
            Type::Numeric(NumericType::Unsigned { bit_size: 1 }),
            BinaryOp::And,
        ) => true,
        (_, _, _) => false,
    }
}

fn array_to_expr(
    array_id: &ValueId,
    array_values: &im::Vector<ValueId>,
    noir_type: &Type,
    dfg: &DataFlowGraph,
) -> Expr {
    let vals_to_expr: Vec<Expr> =
        array_values.iter().map(|val_id| ssa_value_to_expr(val_id, dfg)).collect();
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

fn param_to_expr(value_id: &ValueId, position: usize, noir_type: &Type) -> Expr {
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
    let const_big_int: BigInt = BigInt::from_biguint(num_bigint::Sign::Plus, const_big_uint); //Sign::NoSign??

    SpannedTyped::new(
        &empty_span(), //TODO maybe dont use empty span
        &from_noir_type(noir_type.clone(), None),
        ExprX::Const(Constant::Int(const_big_int)),
    )
}

fn ssa_value_to_expr(value_id: &ValueId, dfg: &DataFlowGraph) -> Expr {
    let value_id = &dfg.resolve(*value_id);
    let value = &dfg[*value_id];
    match value {
        Value::Instruction { instruction: _, position, typ } => {
            param_to_expr(value_id, *position, typ)
        }
        Value::Param { block: _, position, typ } => param_to_expr(value_id, *position, typ),
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
        1 => Some(ssa_value_to_expr(&return_values_ids[0], dfg)),
        _ => {
            let tuple_exprs: Exprs = Arc::new(
                return_values_ids.iter().map(|val_id| ssa_value_to_expr(val_id, dfg)).collect(),
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

fn binary_op_to_vir_binary_op(binary: &BinaryOp, mode: Mode) -> VirBinaryOp {
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
        BinaryOp::Shl => todo!(), // Needs argument bitwidth. Get it here as Optional arg, perhaps
        BinaryOp::Shr => todo!(), // Needs argument bitwidth. Get it here as Optional arg, perhaps
    }
}

fn binary_instruction_to_expr(
    instruction_id: InstructionId,
    binary: &Binary,
    mode: Mode,
    dfg: &DataFlowGraph,
) -> Expr {
    let Binary { lhs, rhs, operator } = binary;
    let lhs_expr = ssa_value_to_expr(lhs, dfg);
    let rhs_expr = ssa_value_to_expr(rhs, dfg);
    let mut binary_exprx =
        ExprX::Binary(binary_op_to_vir_binary_op(operator, mode), lhs_expr.clone(), rhs_expr.clone());
    //Special case for multiplications of booleans
    if is_multiplication_between_bools(lhs, rhs, operator, dfg) {
        binary_exprx = ExprX::Binary(VirBinaryOp::And, lhs_expr.clone(), rhs_expr.clone())
    }
    //Special case for logical or of booleans
    if is_or_between_bools(lhs, rhs, operator, dfg) {
        binary_exprx = ExprX::Binary(VirBinaryOp::Or, lhs_expr.clone(), rhs_expr.clone())
    }
    //Special case for logical and of booleans
    if is_and_between_bools(lhs, rhs, operator, dfg) {
        binary_exprx = ExprX::Binary(VirBinaryOp::And, lhs_expr, rhs_expr)
    }
    SpannedTyped::new(
        &build_span(&instruction_id, format!("lhs({}) binary_op({}) rhs({})", lhs, operator, rhs)),
        &instr_res_type_to_vir_type(binary.result_type(), dfg),
        binary_exprx,
    )
}
/// Depending on the bit width size we want to either return a
/// `unary boolean not` expression or a `unary bit not` expression.
fn bitwise_not_instr_to_exprx(
    value_id: &ValueId,
    dfg: &DataFlowGraph,
    bit_width: Option<IntegerTypeBitwidth>,
) -> ExprX {
    let expr = ssa_value_to_expr(value_id, dfg);

    match bit_width {
        Some(IntegerTypeBitwidth::Width(1)) => ExprX::Unary(UnaryOp::Not, expr),
        Some(width) => ExprX::Unary(UnaryOp::BitNot(Some(width)), expr),
        None => ExprX::Unary(UnaryOp::BitNot(None), expr),
    }
}

fn bitwise_not_instr_to_expr(value_id: &ValueId, dfg: &DataFlowGraph) -> Expr {
    let value = &dfg[*value_id];
    let bit_width: Option<IntegerTypeBitwidth> = match value.get_type() {
        Type::Numeric(numeric_type) => get_integer_bit_width(*numeric_type),
        _ => panic!("Bitwise negation on a non numeric type"),
    };
    let bitnot_exprx = bitwise_not_instr_to_exprx(value_id, dfg, bit_width);
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

fn cast_bool_to_integer(value_id: &ValueId, noir_type: &Type, dfg: &DataFlowGraph) -> Expr {
    let if_return_type = from_noir_type(noir_type.clone(), None);
    let condition = ssa_value_to_expr(value_id, dfg);

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
) -> Expr {
    let cast_exprx = ExprX::Unary(
        UnaryOp::Clip { range: get_int_range(*numeric_type), truncate: false },
        ssa_value_to_expr(value_id, dfg),
    );
    SpannedTyped::new(
        &build_span(value_id, format!("Cast({}) to type({})", value_id, noir_type)),
        &from_noir_type(noir_type.clone(), None),
        cast_exprx,
    )
}

fn cast_instruction_to_expr(value_id: &ValueId, noir_type: &Type, dfg: &DataFlowGraph) -> Expr {
    match dfg[*value_id].get_type() {
        Type::Numeric(NumericType::Unsigned { bit_size: 1 }) => {
            cast_bool_to_integer(value_id, noir_type, dfg)
        }
        Type::Numeric(numeric_type) => {
            cast_integer_to_integer(value_id, noir_type, dfg, numeric_type)
        }
        _ => panic!("Expected that all SSA casts have numeric targets"),
    }
}

fn range_limit_to_expr(
    value_id: &ValueId,
    target_bit_size: u32,
    truncate: bool,
    dfg: &DataFlowGraph,
) -> Expr {
    let value_type = dfg[*value_id].get_type();
    let clip_exprx = match value_type {
        Type::Numeric(numeric_type) => ExprX::Unary(
            UnaryOp::Clip {
                range: trunc_target_int_range(numeric_type, target_bit_size),
                truncate,
            },
            ssa_value_to_expr(value_id, dfg),
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
) -> Expr {
    let binary_equals_expr = SpannedTyped::new(
        &build_span(&instruction_id, format!("lhs({}) == rhs({})", lhs, rhs)),
        &Arc::new(TypX::Bool),
        ExprX::Binary(
            VirBinaryOp::Eq(Mode::Spec), // Verus uses Spec for Eq expressions in asserts
            ssa_value_to_expr(lhs, dfg),
            ssa_value_to_expr(rhs, dfg),
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
) -> Expr {
    let value = &dfg[*value_id];
    let func_id = match value {
        Value::Function(func_id) => func_id,
        _ => unreachable!("You can only call functions in SSA"),
    };

    let name = func_id_into_funx_name(*func_id);
    let arguments_as_expr: Exprs =
        Arc::new(arguments.iter().map(|val_id| ssa_value_to_expr(val_id, dfg)).collect());
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
    dfg: &DataFlowGraph,
) -> Expr {
    let vstd_krate = Some(Arc::new("vstd".to_string()));
    let array_return_type: Typ =
        get_function_ret_type(dfg.instruction_results(instruction_id), dfg);

    let array_length = dfg.try_get_array_length(*array_id).unwrap();
    let array_length_as_type = into_vir_const_int(array_length);
    let typs_for_vstd_func_call: Typs =
        Arc::new(vec![array_return_type.clone(), array_length_as_type.clone()]);
    let array_as_vir_expr: Expr = SpannedTyped::new(
        &build_span(array_id, format!("Array{} as expr", array_id)),
        &Arc::new(TypX::Decorate(
            TypDecoration::Ref,
            None,
            Arc::new(TypX::Primitive(
                Primitive::Array,
                Arc::new(vec![array_return_type.clone(), array_length_as_type.clone()]),
            )),
        )),
        ExprX::Var(id_into_var_ident(*array_id)),
    );
    let index_as_vir_expr = ssa_value_to_expr(index, dfg);

    let segments: Idents =
        Arc::new(vec![Arc::new("array".to_string()), Arc::new("array_index_get".to_string())]);

    let array_get_vir_exprx: ExprX = ExprX::Call(
        CallTarget::Fun(
            CallTargetKind::Static,
            Arc::new(FunX { path: Arc::new(PathX { krate: vstd_krate, segments: segments }) }),
            typs_for_vstd_func_call,
            Arc::new(vec![]),
            AutospecUsage::Final, // Verus uses this flag
        ),
        Arc::new(vec![array_as_vir_expr, index_as_vir_expr]),
    );
    let ref_wrapped_array_return_type: Typ =
        Arc::new(TypX::Decorate(TypDecoration::Ref, None, array_return_type));

    SpannedTyped::new(
        &build_span(&instruction_id, format!("Array get with index {}", index)),
        &ref_wrapped_array_return_type,
        array_get_vir_exprx,
    )
}

fn instruction_to_expr(
    instruction_id: InstructionId,
    instruction: &Instruction,
    mode: Mode,
    dfg: &DataFlowGraph,
) -> Expr {
    match instruction {
        Instruction::Binary(binary) => binary_instruction_to_expr(instruction_id, binary, mode, dfg),
        Instruction::Cast(val_id, noir_type) => cast_instruction_to_expr(val_id, noir_type, dfg),
        Instruction::Not(val_id) => bitwise_not_instr_to_expr(val_id, dfg),
        Instruction::Truncate { value: val_id, bit_size, max_bit_size: _ } => {
            range_limit_to_expr(val_id, *bit_size, true, dfg)
        }
        Instruction::Constrain(lhs, rhs, _) => {
            constrain_instruction_to_expr(instruction_id, lhs, rhs, dfg)
        }
        Instruction::RangeCheck { value: val_id, max_bit_size, assert_message: _ } => {
            range_limit_to_expr(val_id, *max_bit_size, false, dfg)
        }
        Instruction::Call { func, arguments } => {
            call_instruction_to_expr(instruction_id, func, arguments, dfg)
        }
        Instruction::Allocate => unreachable!(), // Optimized away
        Instruction::Load { address: _ } => unreachable!(), // Optimized away
        Instruction::Store { address: _, value: _ } => unreachable!(), // Optimized away
        Instruction::EnableSideEffectsIf { condition: _ } => todo!(), //TODO(totel) Support for mutability
        Instruction::ArrayGet { array, index } => {
            array_get_to_expr(array, index, instruction_id, dfg)
        }
        Instruction::ArraySet { array: _, index: _, value: _, mutable: _ } => todo!(),
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

fn lhs_value_to_pattern(value_id: &ValueId, dfg: &DataFlowGraph) -> Pattern {
    let value_patternx = PatternX::Var { name: id_into_var_ident(*value_id), mutable: false }; // Mutability not supported for the prototype
    SpannedTyped::new(
        &build_span(value_id, format!("Lhs value({})", value_id)),
        &from_noir_type(dfg[*value_id].get_type().clone(), None),
        value_patternx,
    )
}

/// Perhaps needed for tuples. Must be tested explicitly
fn lhs_values_to_pattern(
    lhs_values: &[Id<Value>],
    dfg: &DataFlowGraph,
    instruction_id: Id<Instruction>,
) -> Pattern {
    // I am not sure if the code below works
    // TODO(totel) Test if we ever reach multiple return values
    let tuple_count = lhs_values.len();
    // TODO(totel) I have no idea what binders are
    let binders: Binders<Pattern> = Arc::new(vec![]);
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
        1 => lhs_value_to_pattern(&lhs_ids[0], dfg),
        _ => lhs_values_to_pattern(lhs_ids, dfg, instruction_id),
    }
}

fn instruction_to_stmt(
    instruction: &Instruction,
    dfg: &DataFlowGraph,
    instruction_id: Id<Instruction>,
    mode: Mode,
) -> Stmt {
    let instruction_span =
        build_span(&instruction_id, format!("Instruction({}) statement", instruction_id));

    match dfg.instruction_results(instruction_id).len() {
        0 => Spanned::new(
            build_span(&instruction_id, format!("Instruction({})", instruction_id)),
            StmtX::Expr(instruction_to_expr(instruction_id, instruction, mode, dfg)),
        ),
        _ => Spanned::new(
            instruction_span,
            StmtX::Decl {
                pattern: instruction_to_pattern(instruction_id, dfg),
                mode: Some(Mode::Exec),
                init: Some(instruction_to_expr(instruction_id, instruction, mode, dfg)),
            },
        ),
    }
}

fn is_instruction_enable_side_effects(instruction_id: &InstructionId, dfg: &DataFlowGraph) -> bool {
    match dfg[*instruction_id] {
        Instruction::EnableSideEffectsIf { condition: _ } => true,
        _ => false,
    }
}

/// Returns a SSA block as an expression and
/// the type of the SSA block's terminating instruction
fn basic_block_to_exprx(basic_block_id: Id<BasicBlock>, dfg: &DataFlowGraph) -> (ExprX, Typ) {
    let basic_block = dfg[basic_block_id].clone();
    let mut vir_statements: Vec<Stmt> = Vec::new();

    for instruction_id in basic_block.instructions() {
        if !is_instruction_enable_side_effects(instruction_id, dfg) {
            let statement = instruction_to_stmt(&dfg[*instruction_id], dfg, *instruction_id, Mode::Exec);
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

fn func_body_to_vir_expr(func: &Function) -> Expr {
    let (block_exprx, block_type) = basic_block_to_exprx(func.entry_block(), &func.dfg);
    SpannedTyped::new(
        &build_span(&func.id(), format!("Function's({}) basic block body", func.id())),
        &block_type,
        block_exprx,
    )
}

fn func_attributes_to_vir_expr(
    attribute_instructions: Vec<(Id<Instruction>, Instruction)>,
    dfg: &DataFlowGraph,
) -> Vec<Expr> {
    if let Some((last_instruction_id, last_instruction)) = attribute_instructions.last() {
        let mut vir_statements: Vec<Stmt> = Vec::new();

        for (instruction_id, instruction) in attribute_instructions.clone() {
            let statement = instruction_to_stmt(&instruction, dfg, instruction_id, Mode::Spec);
            vir_statements.push(statement);
        }
        let last_expr = instruction_to_expr(last_instruction_id.clone(), last_instruction, Mode::Spec, dfg);
        vec![SpannedTyped::new(
            &build_span(last_instruction_id, "Formal verification expression".to_string()),
            &get_function_ret_type(dfg.instruction_results(*last_instruction_id), dfg),
            ExprX::Block(Arc::new(vir_statements), Some(last_expr)),
        )]
    } else {
        vec![]
    }
}

fn func_requires_to_vir_expr(func: &Function) -> Exprs {
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
    Arc::new(func_attributes_to_vir_expr(attr_instrs, &func.dfg))
}

fn func_ensures_to_vir_expr(func: &Function) -> Exprs {
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
    Arc::new(func_attributes_to_vir_expr(attr_instrs, &func.dfg))
}

fn build_funx(
    func_id: FunctionId,
    func: &Function,
    current_module: Module,
) -> Result<FunctionX, BuildingKrateError> {
    let function_params = get_function_params(func)?;

    let funx: FunctionX = FunctionX {
        name: func_id_into_funx_name(func_id),
        proxy: None, // No clue. In Verus documentation it says "Proxy used to declare the spec of this function"
        kind: get_func_kind(func), // As far as I understand all functions in SSA are of FunctionKind::Static
        visibility: Visibility { restricted_to: None }, // None is for functions with public visibility. There is no information if the current function is public or private.
        owning_module: Some(current_module.x.path.clone()),
        mode: Mode::Exec, // Currently all functions are Exec. In the near future we will support ghost functions.
        fuel: 1, // In Verus' documentation it says that 1 means visible. I don't understand visible to what exactly
        typ_params: empty_vec_idents(), // There are no generics in SSA
        typ_bounds: empty_vec_generic_bounds(), // There are no generics in SSA
        params: function_params.clone(),
        ret: get_function_return_param(func)?,
        require: func_requires_to_vir_expr(func), // TODO(totel)
        ensure: func_ensures_to_vir_expr(func),   // TODO(totel)
        decrease: Arc::new(vec![]),               // No such feature in the prototype
        decrease_when: None,                      // No such feature in the prototype
        decrease_by: None,                        // No such feature in the prototype
        fndef_axioms: None,                       // Not sure what it is
        mask_spec: None,                          // Not sure what it is
        unwind_spec: None, // To be able to use functions from VSTD we need None on unwinding
        item_kind: ItemKind::Function,
        publish: None, // Only if we use None we pass Verus checks.
        attrs: build_default_funx_attrs(function_params.is_empty()),
        body: Some(func_body_to_vir_expr(func)), // Functions in SSA always have a boyd
        extra_dependencies: vec![],              // Not needed for the prototype
        ens_has_return: is_function_return_void(func), // Should be true if the function returns a value
        returns: None, // SSA functions (I believe) always return values and never expressions. They could also return zero values.
    };
    Ok(funx)
}

pub(crate) fn build_krate(ssa: Ssa) -> Result<Krate, BuildingKrateError> {
    let mut vir: KrateX = KrateX {
        functions: Vec::new(),
        reveal_groups: Vec::new(),
        datatypes: Vec::new(),
        traits: Vec::new(),
        trait_impls: Vec::new(),
        assoc_type_impls: Vec::new(),
        modules: Vec::new(),
        external_fns: Vec::new(),
        external_types: Vec::new(),
        path_as_rust_names: vir::ast_util::get_path_as_rust_names_for_krate(&Arc::new(
            vir::def::VERUSLIB.to_string(),
        )),
        arch: vir::ast::Arch { word_bits: vir::ast::ArchWordBits::Either32Or64 }, // Don't know what bits to use
    };
    let ssa_module = Spanned::new(
        build_span(&Id::<Value>::new(0), format!("SSA module")),
        ModuleX {
            path: Arc::new(PathX {
                krate: None,
                segments: Arc::new(vec![Arc::new(String::from("SSA"))]),
            }),
            reveals: None, //Some(Spanned::new(
                           //     build_span(&Id::<Value>::new(0), format!("SSA module reveals")),
                           //     vir.functions.iter().map(|function| function.x.name.clone()).collect(),
                           // )),
        },
    );
    for (id, func) in &ssa.functions {
        let func_x = build_funx(*id, func, ssa_module.clone())?;
        let function = Spanned::new(
            build_span(id, format!("Function({}) with name {}", id, func.name())),
            func_x,
        );
        vir.functions.push(function);
    }

    vir.modules.push(ssa_module);

    Ok(Arc::new(vir))
}
