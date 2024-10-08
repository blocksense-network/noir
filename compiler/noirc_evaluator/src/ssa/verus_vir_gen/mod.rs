use std::sync::Arc;

use acvm::{AcirField, FieldElement};
use num_bigint::{BigInt, BigUint};
use plonky2::field::types::PrimeField;
use serde::de::value;
use vir::{
    ast::{
        Expr, ExprX, Fun, FunX, FunctionAttrs, FunctionAttrsX, FunctionKind, FunctionX,
        GenericBounds, Idents, IntRange, ItemKind, Krate, KrateX, Mode, Param, ParamX, Params,
        Path, PathX, Pattern, Primitive, SpannedTyped, Stmt, StmtX, Typ, TypDecoration, TypX,
        UnwindSpec, VarIdent, VirErr, Visibility,
    },
    ast_util::air_unique_var,
    def::Spanned,
    messages::Span,
};

use crate::ssa::ir::instruction;

use super::{
    ir::{
        basic_block::BasicBlock,
        dfg::DataFlowGraph,
        function::{Function, FunctionId},
        instruction::{Instruction, TerminatorInstruction},
        map::Id,
        types::{CompositeType, NumericType, Type},
        value::{Value, ValueId},
    },
    ssa_gen::Ssa,
};

/// Should be named differently
pub enum BuildingKrateError {
    SomeError(String),
}

fn func_id_into_segments(function_id: FunctionId) -> Idents {
    Arc::new(vec![Arc::new(function_id.to_usize().to_string())]) // If I use function_id.to_string() it will return "f{id}" instead of only id
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
    Arc::new(TypX::Tuple(Arc::new(Vec::new())))
}

fn empty_vec_idents() -> Idents {
    Arc::new(vec![])
}

fn empty_vec_generic_bounds() -> GenericBounds {
    Arc::new(vec![])
}

fn from_numeric_type(numeric_type: NumericType) -> Typ {
    match numeric_type {
        NumericType::Signed { bit_size } => Arc::new(TypX::Int(IntRange::I(bit_size))),
        NumericType::Unsigned { bit_size } => Arc::new(TypX::Int(IntRange::U(bit_size))),
        NumericType::NativeField => Arc::new(TypX::Int(IntRange::U(FieldElement::max_num_bits()))), // TODO(totel) Decide to what exactly to map Noir Fields to
    }
}

fn into_vir_const_int(number: usize) -> Typ {
    Arc::new(TypX::ConstInt(num_bigint::BigInt::from(number)))
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
        return Arc::new(TypX::Tuple(Arc::new(typs)));
    }
}

/// Maps a Noir type to a Verus VIR type
/// The func_id should be available when the Noir type is Function
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

fn id_into_var_ident(value_id: ValueId) -> VarIdent {
    VarIdent(Arc::new(value_id.to_string()), vir::ast::VarIdentDisambiguate::NoBodyParam)
}

fn empty_var_ident() -> VarIdent {
    VarIdent(Arc::new("empty var ident".to_string()), vir::ast::VarIdentDisambiguate::NoBodyParam)
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
        Span {
            raw_span: Arc::new(()),         // No idea
            id: value_id.to_usize() as u64, // AST id
            data: Vec::new(),               // No idea
            as_string: "param position ".to_owned() + &position.unwrap_or(0).to_string(), // It's used as backup if no other way to show where the error comes from.
        },
        paramx,
    )
}

fn build_empty_param(basic_block_id: Id<BasicBlock>) -> Param {
    let paramx = ParamX {
        name: empty_var_ident(),
        typ: get_empty_vir_type(),
        mode: Mode::Exec,     // For now all parameters are of type Exec
        is_mut: false,        // As far as I understand there is no &mut in SSA
        unwrapped_info: None, // Only if the parameter uses Ghost(x)/Tracked(x) pattern
    };
    Spanned::new(
        Span {
            raw_span: Arc::new(()),               // No idea
            id: basic_block_id.to_usize() as u64, // AST id
            data: Vec::new(),                     // No idea
            as_string: "empty param from basic block ".to_owned()
                + &basic_block_id.to_usize().to_string(), // It's used as backup if no other way to show where the error comes from.
        },
        paramx,
    )
}

fn build_tuple_param_from_values(values: &Vec<ValueId>) -> Param {
    todo!()
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

fn get_function_return_param(func: &Function) -> Result<Param, BuildingKrateError> {
    let entry_block_id = func.entry_block();
    let terminating_instruction = func.dfg[entry_block_id].terminator();

    match terminating_instruction {
        Some(instruction) => match instruction {
            TerminatorInstruction::Return { return_values, call_stack: _ } => {
                if return_values.len() > 1 {
                    // this means that the function either returns a tuple or a struct
                    // this is problematic because tuples and structs are being flatten into
                    // a tuple. What I mean is if we have (A, B, (i32, i32)) where A is a struct
                    // with two Fields and B is a struct with two bools, we will get this as a return value
                    // (Field, Field, bool, bool, i32, i32). A lot of information is lost!!
                    return Ok(build_tuple_param_from_values(return_values));
                }
                if return_values.len() == 0 {
                    return Ok(build_empty_param(entry_block_id));
                }
                let value_id = return_values[0];
                let value = func.dfg[value_id].clone();
                match value {
                    Value::Instruction { instruction: _, position, typ } => {
                        let vir_type = from_noir_type(typ, None);
                        return Ok(build_param(
                            value_id, // air_unique_var(vir::def::RETURN_VALUE)
                            vir_type,
                            Mode::Exec,
                            false,
                            None,
                            Some(position),
                        ));
                    } //TODO Why is an instruction a return value?
                    Value::Param { block: _, position, typ } => {
                        let vir_type = from_noir_type(typ, None);
                        return Ok(build_param(
                            value_id, // air_unique_var(vir::def::RETURN_VALUE)
                            vir_type,
                            Mode::Exec,
                            false,
                            None,
                            Some(position),
                        ));
                    }
                    Value::NumericConstant { constant: _, typ } => {
                        let vir_type = from_noir_type(typ, None);
                        return Ok(build_param(
                            value_id, // air_unique_var(vir::def::RETURN_VALUE)
                            vir_type,
                            Mode::Exec,
                            false,
                            None,
                            None,
                        ));
                    }
                    Value::Array { array: _, typ } => {
                        let vir_type = from_noir_type(typ, None);
                        return Ok(build_param(
                            value_id, // air_unique_var(vir::def::RETURN_VALUE)
                            vir_type,
                            Mode::Exec,
                            false,
                            None,
                            None,
                        ));
                    }
                    Value::Function(func_id) => {
                        let vir_type = from_noir_type(Type::Function, Some(func_id));
                        return Ok(build_param(value_id, vir_type, Mode::Exec, false, None, None));
                    }
                    // TODO(totel) I don't know if those last two ever appear as a return value
                    Value::Intrinsic(intrinsic) => todo!(),
                    Value::ForeignFunction(_) => todo!(),
                }
            }
            _ => return Ok(build_empty_param(entry_block_id)),
        },
        None => {
            return Err(BuildingKrateError::SomeError(
                "Found None as a terminating instruction in a finished SSA block".to_string(),
            ))
        }
    }
}

/// Returns default instance of FunctionAttrs
/// By default we mean the same way a default instance would be
/// constructed in Verus VIR
fn build_default_funx_attrs(zero_args: bool) -> FunctionAttrs {
    Arc::new(FunctionAttrsX {
        uses_ghost_blocks: false,
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

fn terminating_instruction_to_expr(terminating_instruction: &TerminatorInstruction) -> Expr {
    match terminating_instruction {
        TerminatorInstruction::JmpIf {
            condition,
            then_destination,
            else_destination,
            call_stack,
        } => todo!(),
        TerminatorInstruction::Jmp { destination, arguments, call_stack } => todo!(),
        TerminatorInstruction::Return { return_values, call_stack } => todo!(),
    }
}

fn instruction_to_pattern(instruction: &Instruction, dfg: &DataFlowGraph) -> Pattern {
    todo!()
}

fn instruction_to_expr(instruction: &Instruction, dfg: &DataFlowGraph) -> Expr {
    match instruction {
        Instruction::Binary(binary) => todo!(),
        Instruction::Cast(id, _) => todo!(),
        Instruction::Not(id) => todo!(),
        Instruction::Truncate { value, bit_size, max_bit_size } => todo!(),
        Instruction::Constrain(id, id1, constrain_error) => todo!(),
        Instruction::RangeCheck { value, max_bit_size, assert_message } => todo!(),
        Instruction::Call { func, arguments } => todo!(),
        Instruction::Allocate => todo!(),
        Instruction::Load { address } => unreachable!(),
        Instruction::Store { address, value } => unreachable!(),
        Instruction::EnableSideEffectsIf { condition } => todo!(),
        Instruction::ArrayGet { array, index } => todo!(),
        Instruction::ArraySet { array, index, value, mutable } => todo!(),
        Instruction::IncrementRc { value } => todo!(),
        Instruction::DecrementRc { value } => todo!(),
        Instruction::IfElse { then_condition, then_value, else_condition, else_value } => todo!(),
    }
}

fn instruction_to_stmt(
    instruction: &Instruction,
    dfg: &DataFlowGraph,
    instruction_id: Id<Instruction>,
) -> Stmt {
    let instruction_span = Span {
        raw_span: Arc::new(()),
        id: instruction_id.to_usize() as u64,
        data: Vec::new(),
        as_string: "expr for instruction ".to_owned() + &instruction_id.to_usize().to_string(),
    };

    match instruction {
        Instruction::Constrain(id, id1, constrain_error) => {
            Spanned::new(todo!(), StmtX::Expr(todo!()))
        }
        _ => Spanned::new(
            instruction_span,
            StmtX::Decl {
                pattern: instruction_to_pattern(instruction, dfg),
                mode: Some(Mode::Exec),
                init: Some(instruction_to_expr(instruction, dfg)),
            },
        ),
    }
}

fn basic_block_to_exprx(basic_block_id: Id<BasicBlock>, dfg: &DataFlowGraph) -> ExprX {
    let basic_block = dfg[basic_block_id].clone();
    let mut vir_statements: Vec<Stmt> = Vec::new();

    for instruction_id in basic_block.instructions() {
        let statement = instruction_to_stmt(&dfg[*instruction_id], dfg, *instruction_id);
        vir_statements.push(statement);
    }

    assert!(
        basic_block.terminator().is_some(),
        "All finished SSA blocks have a terminating instruction"
    );

    let terminating_instruction = basic_block.terminator().unwrap();
    let block_ending_expr = terminating_instruction_to_expr(terminating_instruction);
    ExprX::Block(Arc::new(vir_statements), Some(block_ending_expr))
    // todo!()
}

fn func_body_to_vir_expr(func: &Function) -> Expr {
    let block_exprx = basic_block_to_exprx(func.entry_block(), &func.dfg);
    SpannedTyped::new(todo!(), todo!(), block_exprx)
    // todo!()
}

fn build_funx(func_id: FunctionId, func: &Function) -> Result<FunctionX, BuildingKrateError> {
    let function_params = get_function_params(func)?;

    let funx: FunctionX = FunctionX {
        name: func_id_into_funx_name(func_id),
        proxy: None, // No clue. In Verus documentation it says "Proxy used to declare the spec of this function"
        kind: get_func_kind(func), // As far as I understand all functions in SSA are of FunctionKind::Static
        visibility: Visibility { restricted_to: None }, // None is for functions with public visibility. There is no information if the current function is public or private.
        owning_module: None,                            // There is no module logic in SSA
        mode: Mode::Exec, // Currently all functions are Exec. In the near future we will support ghost functions.
        fuel: 1, // In Verus' documentation it says that 1 means visible. I don't understand visible to what exactly
        typ_params: empty_vec_idents(), // There are no generics in SSA
        typ_bounds: empty_vec_generic_bounds(), // There are no generics in SSA
        params: function_params.clone(),
        ret: get_function_return_param(func)?,
        require: Arc::new(vec![]),  // TODO(totel)
        ensure: Arc::new(vec![]),   // TODO(totel)
        decrease: Arc::new(vec![]), // No such feature in the prototype
        decrease_when: None,        // No such feature in the prototype
        decrease_by: None,          // No such feature in the prototype
        fndef_axioms: None,         // Not sure what it is
        mask_spec: None,            // Not sure what it is
        unwind_spec: Some(UnwindSpec::NoUnwind),
        item_kind: ItemKind::Function,
        publish: Some(false), // TODO I am not sure if it should be opaque(false) or visible(true)
        attrs: build_default_funx_attrs(function_params.is_empty()),
        body: todo!(),
        extra_dependencies: todo!(),
        ens_has_return: true, // Semantic analysis saves us if the ensures is referencing a unit type
    };
    todo!()
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
        path_as_rust_names: Vec::new(),
        arch: vir::ast::Arch { word_bits: vir::ast::ArchWordBits::Either32Or64 }, // Don't know what bits to use
    };

    for (id, func) in &ssa.functions {
        let func_x = build_funx(*id, func)?;
        let function = Spanned::new(
            Span {
                raw_span: Arc::new(()),                     // No idea
                id: id.to_usize() as u64,                   // AST id
                data: Vec::new(),                           // No idea
                as_string: func.name().to_string().clone(), // It's used as backup if no other way to show where the error comes from.
            },
            func_x,
        );
        vir.functions.push(function); // Note: I assume that functions with no fv attributes will be removed in the SST, AIR or Z3.
                                      // This could be done here if needed.
    }

    let main_function = ssa.functions.into_values().find(|value| value.name() == "main").unwrap();
    let entry_block_id = main_function.entry_block();
    let entry_block = main_function.dfg[entry_block_id].clone();

    // let mut parameters = Vec::new();
    // for value_id in entry_block.parameters().iter() {
    //     self.add_parameter(*value_id)?;
    //     let p2value = self.get(*value_id).unwrap();
    //     match p2value.target.extend_parameter_list(&mut parameters) {
    //         Ok(_) => {}
    //         Err(error) => {
    //             return Err(error.into_runtime_error("parameter list".to_owned(), CallStack::new()));
    //         }
    //     }
    // }
    // for instruction_id in entry_block.instructions() {
    //     match self.add_instruction(*instruction_id) {
    //         Err(error) => {
    //             let instruction = format!("{:?}", self.dfg[*instruction_id].clone());
    //             return Err(
    //                 error.into_runtime_error(instruction, self.dfg.get_call_stack(*instruction_id))
    //             );
    //         }
    //         Ok(_) => (),
    //     }
    // }
    // let mut next_param_idx: usize = 0;
    // for (_, typ, vis) in main_function_signature.0 {
    //     let fields_for_param = typ.field_count() as usize;
    //     if vis == Visibility::Public {
    //         self.asm_writer.register_public_inputs(
    //             &parameters[next_param_idx..next_param_idx + fields_for_param],
    //         );
    //     }
    //     next_param_idx += fields_for_param;
    // }
    // let data = self.asm_writer.move_builder().build::<P2Config>();

    // return Ok(Arc::new(vir));
    todo!()
}
