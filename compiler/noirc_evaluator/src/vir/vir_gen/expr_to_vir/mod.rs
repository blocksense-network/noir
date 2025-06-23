use std::sync::Arc;

use noirc_errors::Location;
use noirc_frontend::monomorphization::ast::{Expression, LValue, Literal};
use vir::ast::VarIdent;

pub mod expr;
pub mod params;
pub mod types;

pub fn ast_var_into_var_ident(name: String, id: u32) -> VarIdent {
    VarIdent(
        Arc::new(name),
        vir::ast::VarIdentDisambiguate::RustcId(
            id.try_into().expect(&format!("Failed to convert u32 {} to usize", id)),
        ),
    )
}

pub fn expression_location(expr: &Expression) -> Option<Location> {
    match expr {
        Expression::Ident(ident) => ident.location,
        Expression::Literal(literal) => literal_location(literal),
        Expression::Block(exprs) => combine_locations(exprs.iter()),
        Expression::Unary(unary) => Some(unary.location),
        Expression::Binary(binary) => Some(binary.location),
        Expression::Index(index) => Some(index.location),
        Expression::Cast(cast) => Some(cast.location),
        Expression::For(for_expr) => {
            let body_location = expression_location(&for_expr.block);
            let range_location = for_expr.start_range_location.merge(for_expr.end_range_location);
            Some(body_location.map_or(range_location, |loc| range_location.merge(loc)))
        }
        Expression::Loop(expr) => expression_location(expr),
        Expression::While(while_expr) => {
            combine_locations([while_expr.condition.as_ref(), while_expr.body.as_ref()].into_iter())
        }
        Expression::If(if_expr) => {
            let expressions = [
                Some(if_expr.condition.as_ref()),
                Some(if_expr.consequence.as_ref()),
                if_expr.alternative.as_deref(),
            ]
            .into_iter()
            .flatten();

            combine_locations(expressions)
        }
        Expression::Match(match_expr) => match_expr
            .cases
            .iter()
            .map(|c| expression_location(&c.branch))
            .chain(match_expr.default_case.as_deref().map(expression_location))
            .flatten()
            .reduce(|a, b| a.merge(b)),
        Expression::Tuple(exprs) => combine_locations(exprs.iter()),
        Expression::ExtractTupleField(expr, _) => expression_location(expr),
        Expression::Call(call) => Some(call.location),
        Expression::Let(let_expr) => expression_location(&let_expr.expression),
        Expression::Constrain(_, loc, _) => Some(*loc),
        Expression::Assign(assign) => {
            let lval_loc = lvalue_location(&assign.lvalue);
            let expr_loc = expression_location(&assign.expression);

            match (lval_loc, expr_loc) {
                (Some(l), Some(r)) => Some(l.merge(r)),
                (Some(l), None) => Some(l),
                (None, Some(r)) => Some(r),
                (None, None) => None,
            }
        }
        Expression::Semi(expr) => expression_location(expr),
        Expression::Clone(expr) => expression_location(expr),
        Expression::Drop(expr) => expression_location(expr),
        Expression::Break | Expression::Continue => None,
        Expression::Quant(_, _, expr) => expression_location(expr),
    }
}

fn combine_locations<'a, I>(exprs: I) -> Option<Location>
where
    I: Iterator<Item = &'a Expression>,
{
    exprs.filter_map(expression_location).reduce(|a, b| a.merge(b))
}

fn lvalue_location(lvalue: &LValue) -> Option<Location> {
    match lvalue {
        LValue::Ident(ident) => ident.location,
        LValue::Index { location, .. } => Some(*location),
        LValue::MemberAccess { object, .. } => lvalue_location(object),
        LValue::Dereference { reference, .. } => lvalue_location(reference),
    }
}

fn literal_location(literal: &Literal) -> Option<Location> {
    match literal {
        Literal::Integer(_, _, location) => Some(*location),
        Literal::Array(array) | Literal::Slice(array) => combine_locations(array.contents.iter()),
        Literal::FmtStr(_, _, expr) => expression_location(expr),
        _ => None,
    }
}
