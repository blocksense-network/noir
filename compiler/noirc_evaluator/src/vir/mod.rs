use noirc_frontend::monomorphization::ast::Program;
use vir::ast::Krate;
use vir_gen::{BuildingKrateError, build_krate};
pub mod vir_gen;

pub fn create_verus_vir(program: Program) -> Result<Krate, BuildingKrateError> {
    // let program = monomorph_ast_optimization_passes(program); // TODO(totel)
    build_krate(program)
}
