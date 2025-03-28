pub mod context;
mod dap;
pub mod errors;
pub mod foreign_calls;
mod repl;
mod source_code_printer;

use std::io::{Read, Write};

use ::dap::errors::ServerError;
use ::dap::server::Server;
use acvm::acir::native_types::{WitnessMap, WitnessStack};
use acvm::{BlackBoxFunctionSolver, FieldElement};

use nargo::NargoError;
use noirc_driver::CompiledProgram;

pub fn run_repl_session<B: BlackBoxFunctionSolver<FieldElement>>(
    solver: &B,
    program: CompiledProgram,
    initial_witness: WitnessMap<FieldElement>,
    raw_source_printing: bool,
) -> Result<Option<WitnessStack<FieldElement>>, NargoError<FieldElement>> {
    repl::run(solver, program, initial_witness, raw_source_printing)
}

pub fn run_dap_loop<R: Read, W: Write, B: BlackBoxFunctionSolver<FieldElement>>(
    server: Server<R, W>,
    solver: &B,
    program: CompiledProgram,
    initial_witness: WitnessMap<FieldElement>,
) -> Result<(), ServerError> {
    dap::run_session(server, solver, program, initial_witness)
}
