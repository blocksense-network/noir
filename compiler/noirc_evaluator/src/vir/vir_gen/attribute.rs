use std::sync::Arc;

use noirc_frontend::monomorphization::ast::Function;
use vir::ast::Exprs;

pub fn func_requires_to_vir_expr(func: &Function) -> Exprs {
    Arc::new(Vec::new()) //TODO(totel)
}

pub fn func_ensures_to_vir_expr(func: &Function) -> Exprs {
    Arc::new(Vec::new()) //TODO(totel)
}
