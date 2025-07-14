use noirc_errors::Location;
use noirc_frontend::monomorphization::ast as mast;
use typing::OptionalType;
use std::{collections::BTreeMap, fmt::Debug};

use crate::{
    ast::{OffsetExpr, SpannedExpr, cata},
    parse::build_location,
};

// NOTE: all types inside are not prefixed, to be used as `ast::OffsetExpr`
pub mod ast;
pub mod parse;
pub mod typing;

#[derive(Debug)]
pub struct State<'a> {
    pub function: &'a mast::Function,
    pub global_constants: &'a BTreeMap<mast::GlobalId, (String, mast::Type, mast::Expression)>,
    pub functions: &'a BTreeMap<mast::FuncId, mast::Function>,
    pub min_local_id: &'a mut u32,
}

#[derive(Debug, Clone)]
pub enum Attribute {
    Ghost,
    Ensures(ast::SpannedExpr),
    Requires(ast::SpannedExpr),
}

#[derive(Debug, Clone)]
pub struct MonomorphizationRequest {
    pub function_identifier: String,
    // NOTE: `None` for untyped integer literals
    pub param_types: Vec<OptionalType>,
}

fn span_expr(annotation_location: Location, full_length: u32, expr: OffsetExpr) -> SpannedExpr {
    cata(expr, &|(prev_offset, after_offset), exprf| SpannedExpr {
        ann: build_location(annotation_location, full_length, prev_offset, after_offset),
        expr: Box::new(exprf),
    })
}
