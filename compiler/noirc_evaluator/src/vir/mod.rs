use noirc_frontend::monomorphization::ast::Program;
use vir::ast::Krate;
use vir_gen::{BuildingKrateError, build_krate};

use crate::vir::opt_passes::monomorph_ast_optimization_passes;
pub mod vir_gen;
pub mod opt_passes;

pub fn create_verus_vir(program: Program) -> Result<Krate, BuildingKrateError> {
    let program = monomorph_ast_optimization_passes(program);
    build_krate(program)
}
