use std::sync::Arc;

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
