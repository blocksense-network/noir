use noirc_errors::{Location, Span};
use noirc_frontend::monomorphization::ast as mast;
use nom::Finish;
use std::{collections::BTreeMap, fmt::Debug};
use vir::messages::Span as VirSpan;

use crate::{
    ast::{OffsetExpr, SpannedExpr, cata},
    parse::{Error as ParseError, build_location, parse_expression_expr, parse_identifier},
};

// NOTE: all types inside are not prefixed, to be used as `ast::OffsetExpr`
pub mod ast;
pub mod parse;
pub mod typing;

#[derive(Debug, Clone)]
pub struct State<'a> {
    pub function: &'a mast::Function,
    pub global_constants: &'a BTreeMap<mast::GlobalId, (String, mast::Type, mast::Expression)>,
    pub functions: &'a BTreeMap<mast::FuncId, mast::Function>,
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
    pub param_types: Vec<Option<mast::Type>>,
}

fn span_expr(annotation_location: Location, full_length: u32, expr: OffsetExpr) -> SpannedExpr {
    cata(expr, &|(prev_offset, after_offset), exprf| SpannedExpr {
        ann: build_location(annotation_location, full_length, prev_offset, after_offset),
        expr: Box::new(exprf),
    })
}

pub fn parse_attribute<'a>(
    annotation: &'a str,
    mut location: Location,
    function: &'a mast::Function,
    global_constants: &'a BTreeMap<mast::GlobalId, (String, mast::Type, mast::Expression)>,
    functions: &'a BTreeMap<mast::FuncId, mast::Function>,
) -> Result<Attribute, ParseError> {
    // NOTE: #['...]
    //       ^^^^^^^ - received `Location`
    //          ^^^  - relevant stuff
    // TODO: don't do this here
    location = Location {
        span: Span::inclusive(location.span.start() + 3, location.span.end() - 1),
        ..location
    };

    let input = annotation;
    let (input, attribute_type) = parse_identifier(input).finish()?; //.map_err(|e| e.parser_errors)?;

    let get_expr = || -> Result<SpannedExpr, ParseError> {
        let (rest, expr) = parse_expression_expr(input).finish()?; //.map_err(|e| e.parser_errors)?;
        assert_eq!(rest, "");
        Ok(span_expr(location, annotation.len() as u32, expr))
    };

    Ok(match attribute_type {
        "ghost" => Attribute::Ghost,
        "ensures" => Attribute::Ensures(get_expr()?),
        "requires" => Attribute::Requires(get_expr()?),
        _ => {
            // return Err(vec![ResolverError::VariableNotDeclared {
            //     name: attribute_type.to_string(),
            //     location,
            // }]);
            return Err(ParseError { parser_errors: vec![] });
        }
    })
}
