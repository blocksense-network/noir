mod source_location;
use source_location::SourceLocation;

mod stack_frame;
use stack_frame::StackFrame;

mod debugger_glue;
use debugger_glue::{get_current_source_locations, get_stack_frames};

pub mod tracer_glue;
use tracer_glue::TracerGlue;

pub mod tail_diff_vecs;
use tail_diff_vecs::tail_diff_vecs;

use acvm::acir::circuit::brillig::BrilligBytecode;
use acvm::{acir::circuit::Circuit, acir::native_types::WitnessMap};
use acvm::{BlackBoxFunctionSolver, FieldElement};
use nargo::NargoError;
use noir_debugger::context::{DebugCommandResult, DebugContext};
use noir_debugger::foreign_calls::DefaultDebugForeignCallExecutor;
use noirc_artifacts::debug::DebugArtifact;
use runtime_tracing::{Line, Tracer};
use std::path::PathBuf;

/// The result from step_debugger: the debugger either paused at a new location, reached the end of
/// execution, or hit some kind of an error. Takes the error type as a parameter.
enum DebugStepResult<Error> {
    /// The debugger reached a new location and the execution is paused at it. The wrapped value is
    /// a vector, because if the next source line is a function call, one debugger step includes
    /// it, together with the first line of the called function. This is just how `nargo debug`
    /// works and a fact of life we choose not to change.
    Paused(Vec<SourceLocation>),
    /// The debuger reached the end of the program and finished execution.
    Finished,
    /// The debugger reached an error and cannot continue.
    Error(Error),
}

pub struct TracingContext<'a, B: BlackBoxFunctionSolver<FieldElement>> {
    debug_context: DebugContext<'a, B>,
    /// The source location at the current moment of tracing.
    source_locations: Vec<SourceLocation>,
    /// The stack trace at the current moment; last call is last in the vector.
    stack_frames: Vec<StackFrame>,
}

impl<'a, B: BlackBoxFunctionSolver<FieldElement>> TracingContext<'a, B> {
    pub fn new(
        blackbox_solver: &'a B,
        circuits: &'a [Circuit<FieldElement>],
        debug_artifact: &'a DebugArtifact,
        initial_witness: WitnessMap<FieldElement>,
        unconstrained_functions: &'a [BrilligBytecode<FieldElement>],
    ) -> Self {
        let foreign_call_executor =
            Box::new(DefaultDebugForeignCallExecutor::from_artifact(true, debug_artifact));
        let debug_context = DebugContext::new(
            blackbox_solver,
            circuits,
            debug_artifact,
            initial_witness.clone(),
            foreign_call_executor,
            unconstrained_functions,
        );

        Self { debug_context, source_locations: vec![], stack_frames: vec![] }
    }

    /// Steps the debugger until a new line is reached, or the debugger returns anything other than
    /// Ok.
    ///
    /// Propagates the debugger result.
    fn step_debugger(&mut self) -> DebugStepResult<NargoError<FieldElement>> {
        loop {
            match self.debug_context.next_into() {
                DebugCommandResult::Done => return DebugStepResult::Finished,
                DebugCommandResult::Error(error) => return DebugStepResult::Error(error),
                DebugCommandResult::BreakpointReached(loc) => {
                    // Note: this is panic! instead of an error, because it is more serious and
                    // indicates an internal inconsistency, rather than a recoverable error.
                    panic!("Error: Breakpoint unexpected in tracer; loc={loc}")
                }
                DebugCommandResult::Ok => (),
            }

            let source_locations = get_current_source_locations(&self.debug_context);
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
    fn update_record(
        &mut self,
        tracer_glue: &mut TracerGlue,
        source_locations: &Vec<SourceLocation>,
    ) {
        let stack_frames = get_stack_frames(&self.debug_context);
        let (first_nomatch, dropped_frames, new_frames) =
            tail_diff_vecs(&self.stack_frames, &stack_frames);

        for _ in dropped_frames {
            tracer_glue.register_return();
        }

        for i in 0..new_frames.len() {
            tracer_glue.register_call(&source_locations[first_nomatch + i], new_frames[i]);
        }

        self.stack_frames = stack_frames;

        let (_, _, new_source_locations) = tail_diff_vecs(&self.source_locations, source_locations);
        for location in new_source_locations {
            tracer_glue.register_step(location);
            if let Some(last_frame) = &self.stack_frames.last() {
                tracer_glue.register_variables(last_frame);
            }
        }
    }
}

pub fn trace_circuit<B: BlackBoxFunctionSolver<FieldElement>>(
    blackbox_solver: &B,
    circuits: &[Circuit<FieldElement>],
    debug_artifact: &DebugArtifact,
    initial_witness: WitnessMap<FieldElement>,
    unconstrained_functions: &[BrilligBytecode<FieldElement>],
    tracer: &mut Tracer,
) -> Result<(), NargoError<FieldElement>> {
    let mut tracing_context = TracingContext::new(
        blackbox_solver,
        circuits,
        debug_artifact,
        initial_witness,
        unconstrained_functions,
    );

    if tracing_context.debug_context.get_current_debug_location().is_none() {
        println!("Warning: circuit contains no opcodes; generating no trace");
        return Ok(());
    }

    let SourceLocation { filepath, line_number } = SourceLocation::create_unknown();
    tracer.start(&PathBuf::from(filepath.to_string()), Line(line_number as i64));
    let mut tracer_glue = TracerGlue::new(tracer);
    loop {
        let source_locations = match tracing_context.step_debugger() {
            DebugStepResult::Finished => break,
            DebugStepResult::Error(err) => {
                println!("Error: {err}");
                break;
            }
            DebugStepResult::Paused(source_location) => source_location,
        };

        tracing_context.update_record(&mut tracer_glue, &source_locations);

        // This update is intentionally explicit here, to show what drives the loop.
        tracing_context.source_locations = source_locations;
    }

    Ok(())
}
