use formal_verification::parse::Attribute;
use noirc_frontend::monomorphization::ast::{FuncId, Program};
use vir::ast::Krate;
use vir_gen::{BuildingKrateError, build_krate};

use crate::vir::{
    opt_passes::monomorph_ast_optimization_passes, vir_gen::build_krate_with_ready_annotations,
};
pub mod opt_passes;
pub mod vir_gen;

pub fn create_verus_vir(program: Program) -> Result<Krate, BuildingKrateError> {
    let program = monomorph_ast_optimization_passes(program);
    build_krate(program)
}

/// Same as `create_verus_vir` but expects the FV attributes
/// to be already transformed into VIR form.
pub fn create_verus_vir_with_ready_annotations(
    program: Program,
    fv_annotations: Vec<(FuncId, Vec<Attribute>)>,
) -> Result<Krate, BuildingKrateError> {
    let program = monomorph_ast_optimization_passes(program);
    build_krate_with_ready_annotations(program, fv_annotations)
}
