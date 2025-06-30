use std::{collections::BTreeMap, sync::Arc};

use noirc_frontend::monomorphization::ast::{Expression, Function, GlobalId, MonomorphizedFvAttribute, Type};
use vir::ast::{Exprs, Mode};

use crate::vir::vir_gen::expr_to_vir::expr::ast_expr_to_vir_expr;

pub fn func_requires_to_vir_expr(
    func: &Function,
    globals: &BTreeMap<GlobalId, (String, Type, Expression)>,
) -> Exprs {
    Arc::new(
        func.formal_verification_attributes
            .iter()
            .filter_map(|attribute| {
                if let MonomorphizedFvAttribute::Requires(expression) = attribute {
                    Some(ast_expr_to_vir_expr(expression, Mode::Spec, globals))
                } else {
                    None
                }
            })
            .collect(),
    )
}

pub fn func_ensures_to_vir_expr(
    func: &Function,
    globals: &BTreeMap<GlobalId, (String, Type, Expression)>,
) -> Exprs {
    Arc::new(
        func.formal_verification_attributes
            .iter()
            .filter_map(|attribute| {
                if let MonomorphizedFvAttribute::Ensures(expression) = attribute {
                    Some(ast_expr_to_vir_expr(expression, Mode::Spec, globals))
                } else {
                    None
                }
            })
            .collect(),
    )
}
