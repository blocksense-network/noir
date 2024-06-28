use noir_debugger::context::{DebugCommandResult, DebugContext};

use acvm::{acir::circuit::Circuit, acir::native_types::WitnessMap};
use acvm::{BlackBoxFunctionSolver, FieldElement};

use acvm::acir::circuit::brillig::BrilligBytecode;

use noir_debugger::foreign_calls::DefaultDebugForeignCallExecutor;
use noirc_artifacts::debug::DebugArtifact;

use fm::PathString;
use std::cmp::min;
use std::path::PathBuf;

use runtime_tracing::{Line, Tracer};

use nargo::NargoError;

#[derive(PartialEq, Debug)]
struct SourceLocation {
    filepath: PathString,
    line_number: isize,
}

impl SourceLocation {
    /// Creates a source location that represents an unknown place in the source code.
    fn create_unknown() -> SourceLocation {
        SourceLocation { filepath: PathString::from_path(PathBuf::from("?")), line_number: -1 }
    }
}

enum DebugStepResult<Error> {
    /// The debugger reached a new location and the execution is paused at it.
    Paused(Vec<SourceLocation>),
    /// The debuger reached the end of the program and finished execution.
    Finished,
    /// The debugger reached an error and cannot continue.
    Error(Error),
}

pub struct TracingContext<'a, B: BlackBoxFunctionSolver<FieldElement>> {
    /// Nested structure, used for extracting tracing information by leveraging the functionality
    /// of `nargo debug`.
    debug_context: DebugContext<'a, B>,
    /// The call stack at the current moment of tracing.
    source_locations: Vec<SourceLocation>,
}

impl<'a, B: BlackBoxFunctionSolver<FieldElement>> TracingContext<'a, B> {
    pub fn new(
        blackbox_solver: &'a B,
        circuit: &'a Circuit<FieldElement>,
        debug_artifact: &'a DebugArtifact,
        initial_witness: WitnessMap<FieldElement>,
        unconstrained_functions: &'a [BrilligBytecode<FieldElement>],
    ) -> Self {
        let foreign_call_executor =
            Box::new(DefaultDebugForeignCallExecutor::from_artifact(true, debug_artifact));
        let debug_context = DebugContext::new(
            blackbox_solver,
            circuit,
            debug_artifact,
            initial_witness.clone(),
            foreign_call_executor,
            unconstrained_functions,
        );

        Self { debug_context, source_locations: vec![] }
    }

    /// Extracts the stack of current source locations from the debugger, given that the
    /// relevant debugging information is present. In the context of this method, a source location
    /// is a path to a source file and a line in that file. The most recently called function is
    /// last in the returned vector/stack.
    ///
    /// Otherwise, returns a &str containing a warning message about the missing data.
    fn get_current_source_locations(&self) -> Vec<SourceLocation> {
        let call_stack = self.debug_context.get_call_stack();

        let mut result: Vec<SourceLocation> = vec![];
        for opcode_location in call_stack {
            let locations =
                self.debug_context.get_source_location_for_opcode_location(&opcode_location);
            for source_location in locations {
                let filepath = match self.debug_context.get_filepath_for_location(source_location) {
                    Ok(filepath) => filepath,
                    Err(error) => {
                        println!("Warning: could not get filepath for source location: {error}");
                        result.push(SourceLocation::create_unknown());
                        continue;
                    }
                };

                let line_number = match self.debug_context.get_line_for_location(source_location) {
                    Ok(line) => line as isize + 1,
                    Err(error) => {
                        println!("Warning: could not get line for source location: {error}");
                        result.push(SourceLocation::create_unknown());
                        continue;
                    }
                };

                result.push(SourceLocation { filepath, line_number })
            }
        }

        result
    }

    /// Steps the debugger until a new line is reached, or the debugger returns anything other than
    /// Ok. If there are no source locations associated with the opcodes at any time, the stepping
    /// is also continued.
    ///
    /// Returns a dedicated type, which on success contains the call stack at which debugging has
    /// paused.
    fn step_debugger(&mut self) -> DebugStepResult<NargoError<FieldElement>> {
        loop {
            match self.debug_context.next_into() {
                DebugCommandResult::Done => return DebugStepResult::Finished,
                DebugCommandResult::Error(error) => return DebugStepResult::Error(error),
                DebugCommandResult::BreakpointReached(loc) => {
                    panic!("Error: Breakpoint unexpected in tracer; loc={loc}")
                }
                DebugCommandResult::Ok => (),
            }

            let source_locations = self.get_current_source_locations();
            if source_locations.is_empty() {
                println!("Warning: no call stack");
                continue;
            };

            if self.source_locations.len() == source_locations.len()
                && self.source_locations.last().unwrap() == source_locations.last().unwrap()
            {
                // Continue stepping until a new line in the same file is reached, or the current file
                // has changed.
                // TODO(coda-bug/r916): a function call could result in an extra step
                continue;
            }

            return DebugStepResult::Paused(source_locations);
        }
    }

    /// Propagates information about the current execution state to `tracer`.
    fn update_record(&mut self, tracer: &mut Tracer, source_locations: Vec<SourceLocation>) {
        // Find the last index of the previous and current stack traces, until which they are
        // identical.
        let mut last_match: isize = -1;
        for i in 0..min(self.source_locations.len(), source_locations.len()) {
            if self.source_locations[i] == source_locations[i] {
                last_match = i as isize;
                continue;
            }
            break;
        }
        // For the rest of the indexes of the new call stack: register a step that was performed to
        // reach that frame of the call stack.
        for i in ((last_match + 1) as usize)..source_locations.len() {
            let SourceLocation { filepath, line_number } = &source_locations[i];
            tracer.register_step(&PathBuf::from(filepath.to_string()), Line(*line_number as i64));
        }
        self.source_locations = source_locations;
    }
}

pub fn trace_circuit<B: BlackBoxFunctionSolver<FieldElement>>(
    blackbox_solver: &B,
    circuit: &Circuit<FieldElement>,
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

    if tracing_context.debug_context.get_current_opcode_location().is_none() {
        println!("Warning: circuit contains no opcodes; generating no trace");
        return Ok(());
    }

    let SourceLocation { filepath, line_number } = SourceLocation::create_unknown();
    tracer.start(&PathBuf::from(filepath.to_string()), Line(line_number as i64));
    loop {
        let source_locations = match tracing_context.step_debugger() {
            DebugStepResult::Finished => break,
            DebugStepResult::Error(err) => {
                println!("Error: {err}");
                break;
            }
            DebugStepResult::Paused(source_locations) => source_locations,
        };

        tracing_context.update_record(tracer, source_locations);
    }

    Ok(())
}
