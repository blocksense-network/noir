use noirc_frontend::monomorphization::ast::Program;

use crate::vir::opt_passes::tuple_deconstruction::fix_tuple_deconstruction_pass;
pub mod tuple_deconstruction;


pub fn monomorph_ast_optimization_passes(mut program: Program) -> Program {
    fix_tuple_deconstruction_pass(&mut program);
    program
}