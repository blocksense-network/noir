use std::sync::Arc;

use noirc_frontend::monomorphization::ast::{Function, MonomorphizedFvAttribute};
use vir::ast::{Exprs, Mode};

use crate::vir::vir_gen::expr_to_vir::expr::ast_expr_to_vir_expr;

pub fn func_requires_to_vir_expr(func: &Function) -> Exprs {
    Arc::new(
        func.formal_verification_attributes
            .iter()
            .filter_map(|attribute| {
                if let MonomorphizedFvAttribute::Requires(expression) = attribute {
                    Some(ast_expr_to_vir_expr(expression, Mode::Spec))
                } else {
                    None
                }
            })
            .collect(),
    )
}

pub fn func_ensures_to_vir_expr(func: &Function) -> Exprs {
    Arc::new(
        func.formal_verification_attributes
            .iter()
            .filter_map(|attribute| {
                if let MonomorphizedFvAttribute::Ensures(expression) = attribute {
                    Some(ast_expr_to_vir_expr(expression, Mode::Spec))
                } else {
                    None
                }
            })
            .collect(),
    )
}
