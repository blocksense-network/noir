use std::sync::Arc;

use vir::ast::VarIdent;

pub mod expr;
pub mod params;
pub mod types;

pub fn id_into_var_ident(id: u32) -> VarIdent {
    VarIdent(
        Arc::new(id.to_string()),
        vir::ast::VarIdentDisambiguate::RustcId(
            id.try_into().expect(&format!("Failed to convert u32 {} to usize", id)),
        ),
    )
}
