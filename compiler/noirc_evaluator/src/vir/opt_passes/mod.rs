use noirc_frontend::monomorphization::ast::Program;

use crate::vir::opt_passes::{loop_unroll::unroll_for_loops_pass, tuple_deconstruction::fix_tuple_deconstruction_pass};
pub mod loop_unroll;
pub mod tuple_deconstruction;

pub fn monomorph_ast_optimization_passes(mut program: Program) -> Program {
    fix_tuple_deconstruction_pass(&mut program);
    unroll_for_loops_pass(&mut program);
    program
}
