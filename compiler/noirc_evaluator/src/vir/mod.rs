use noirc_frontend::monomorphization::ast::Program;
use vir::ast::Krate;
use crate::errors::RuntimeError;

pub fn create_verus_vir(
    program: Program,
) -> Result<Krate, RuntimeError> {
    todo!()
}