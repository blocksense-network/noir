use acvm::acir::native_types::WitnessStack;
use noir_debugger::context::{DebugCommandResult, DebugContext};

use acvm::{acir::circuit::Circuit, acir::native_types::WitnessMap};
use acvm::{BlackBoxFunctionSolver, FieldElement};

use acvm::acir::circuit::brillig::BrilligBytecode;

use noir_debugger::foreign_calls::DefaultDebugForeignCallExecutor;
use noirc_artifacts::debug::DebugArtifact;

use fm::PathString;
use std::path::PathBuf;

use runtime_tracing::{Line, Tracer};

use nargo::NargoError;

pub struct TracingContext<'a, B: BlackBoxFunctionSolver<FieldElement>> {
    debug_context: DebugContext<'a, B>,
    current_filepath: PathString, // The path to the file currently pointed at by debugger.
    current_line: isize,          // The line in the source code that the debugger has reached.
}

impl<'a, B: BlackBoxFunctionSolver<FieldElement>> TracingContext<'a, B> {
    pub fn new(
        blackbox_solver: &'a B,
        circuit: &'a [Circuit<FieldElement>],
        debug_artifact: &'a DebugArtifact,
        initial_witness: WitnessMap<FieldElement>,
        unconstrained_functions: &'a [BrilligBytecode<FieldElement>],
    ) -> Self {
        let foreign_call_executor =
            Box::new(DefaultDebugForeignCallExecutor::from_artifact(nargo::PrintOutput::Stdout, debug_artifact));
        let debug_context = DebugContext::new(
            blackbox_solver,
            circuit,
            debug_artifact,
            initial_witness.clone(),
            foreign_call_executor,
            unconstrained_functions,
        );

        Self {
            debug_context,
            current_filepath: PathString::from_path(PathBuf::new()),
            current_line: -1isize,
        }
    }

    /// Steps the debugger until a new line is reached, or the debugger returns anything other than
    /// Ok.
    ///
    /// Propagates the debugger result.
    fn step_debugger(&mut self) -> DebugCommandResult {
        loop {
            let result = self.debug_context.next_into();

            match &result {
                DebugCommandResult::Done
                | DebugCommandResult::Error(_)
                | DebugCommandResult::BreakpointReached(_) => return result,
                DebugCommandResult::Ok => (),
            }

            let call_stack = self.debug_context.get_call_stack();
            let opcode_location = match call_stack.last() {
                Some(location) => location,
                None => {
                    println!("Warning: no call stack");
                    continue;
                }
            };

            let locations =
                self.debug_context.get_source_location_for_debug_location(opcode_location);
            let source_location = match locations.last() {
                Some(location) => location,
                None => {
                    println!("Warning: no source location mapped to opcode");
                    continue;
                }
            };

            let filepath = match self.debug_context.get_filepath_for_location(*source_location) {
                Ok(filepath) => filepath,
                Err(error) => {
                    println!("Warning: could not get filepath for source location: {error}");
                    continue;
                }
            };

            let current_line = match self.debug_context.get_line_for_location(*source_location) {
                Ok(line) => line as isize + 1,
                Err(error) => {
                    println!("Warning: could not get line for source location: {error}");
                    continue;
                }
            };

            if self.current_filepath == filepath && self.current_line == current_line {
                // Continue stepping until a new line in the same file is reached, or the current file
                // has changed.
                //
                // TODO(stanm): address potential bug, where a function call could switch file; switch
                // back; and get an extra step on the original line containing the call.
                continue;
            }

            self.current_filepath = filepath;
            self.current_line = current_line;
            return result;
        }
    }

    /// Propagates information about the current execution state to `tracer`.
    fn update_record(&mut self, tracer: &mut Tracer) {
        tracer.register_step(
            &PathBuf::from(self.current_filepath.to_string()),
            Line(self.current_line as i64),
        );
    }
}

pub fn trace_circuit<B: BlackBoxFunctionSolver<FieldElement>>(
    blackbox_solver: &B,
    circuit: &[Circuit<FieldElement>],
    debug_artifact: &DebugArtifact,
    initial_witness: WitnessMap<FieldElement>,
    unconstrained_functions: &[BrilligBytecode<FieldElement>],
    tracer: &mut Tracer,
) -> Result<(), NargoError<FieldElement>> {
    let mut tracing_context = TracingContext::new(
        blackbox_solver,
        circuit,
        debug_artifact,
        initial_witness,
        unconstrained_functions,
    );

    if tracing_context.debug_context.get_current_debug_location().is_none() {
        println!("Warning: circuit contains no opcodes; generating no trace");
        return Ok(());
    }

    let mut steps = 0;
    tracer.start(&PathBuf::from(""), Line(-1));
    loop {
        match tracing_context.step_debugger() {
            DebugCommandResult::Done => break,
            DebugCommandResult::Ok => steps += 1,
            DebugCommandResult::Error(err) => {
                println!("Error: {err}");
                break;
            }
            DebugCommandResult::BreakpointReached(loc) => {
                panic!("Error: Breakpoint unexpected in tracer; loc={loc}");
            }
        }

        tracing_context.update_record(tracer);
    }
    println!("Total tracing steps: {steps}");

    Ok(())
}
