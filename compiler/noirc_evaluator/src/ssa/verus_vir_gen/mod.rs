mod attributes;
mod context;
mod expr_to_vir;
mod function;

use acvm::{AcirField, FieldElement};
use context::{ResultIdFixer, SSAContext};
use function::build_funx;
use num_bigint::{BigInt, BigUint};
use std::sync::Arc;
use vir::{
    ast::{
        ArithOp, AutospecUsage, BitwiseOp, CallTarget, CallTargetKind, Constant, Dt, Expr, ExprX,
        Exprs, Fun, FunX, FunctionKind, Idents, ImplPath, ImplPaths, InequalityOp, IntRange,
        IntegerTypeBitwidth, Krate, KrateX, Mode, ModuleX, ParamX, Params, PathX, Pattern,
        PatternX, Primitive, SpannedTyped, Stmt, StmtX, Typ, TypDecoration, TypX, Typs, UnaryOp,
        VarIdent,
    },
    ast_util::mk_tuple,
    def::{prefix_tuple_variant, Spanned},
    messages::Span,
};

use vir::ast::BinaryOp as VirBinaryOp;

use super::{
    ir::{
        basic_block::BasicBlock,
        dfg::{CallStack, DataFlowGraph},
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

/// This function is technical debt. Its use should be minimized
fn empty_span() -> Span {
    Span { raw_span: Arc::new(()), id: 0, data: vec![], as_string: String::new() }
}

fn encode_span_to_string(call_stack: CallStack) -> String {
    if let Some(last_call) = call_stack.last() {
        let stringified_span: String =
            last_call.span.start().to_string() + ", " + &last_call.span.end().to_string();
        let stringified_file_id: String = last_call.file.as_usize().to_string();
        String::from("(") + &stringified_span + ", " + &stringified_file_id + ")"
    } else {
        String::new()
    }
}

fn build_span<A>(ast_id: &Id<A>, debug_string: String, span: Option<CallStack>) -> Span {
    let encoded_span =
        if let Some(call_stack) = span { encode_span_to_string(call_stack) } else { String::new() };
    Span {
        raw_span: Arc::new(()),       // Currently unusable because of unknown bug
        id: ast_id.to_usize() as u64, // AST id
        data: Vec::new(),             // No idea
        as_string: encoded_span + &debug_string, // It's used as backup if there is no other way to show where the error comes from.
    }
}

fn empty_vec_idents() -> Idents {
    Arc::new(vec![])
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
        build_span(&Id::<Value>::new(0), format!("SSA module"), None),
        ModuleX {
            path: Arc::new(PathX {
                krate: None,
                segments: Arc::new(vec![Arc::new(String::from("SSA"))]),
            }),
            reveals: None,
        },
    );
    for (id, func) in &ssa.functions {
        let func_x = build_funx(*id, func, ssa_module.clone())?;
        let function = Spanned::new(
            build_span(id, format!("Function({}) with name {}", id, func.name()), None), //TODO See if we could get the function's span
            func_x,
        );
        vir.functions.push(function);
    }

    vir.modules.push(ssa_module);

    Ok(Arc::new(vir))
}
