use noir_debugger::context::{DebugCommandResult, DebugContext};

use acvm::{acir::circuit::Circuit, acir::native_types::WitnessMap};
use acvm::{AcirField, BlackBoxFunctionSolver, FieldElement};

use acvm::acir::circuit::brillig::BrilligBytecode;

use nargo::artifacts::debug::DebugArtifact;
use nargo::artifacts::trace::{
    CallRecord, StepRecord, TraceArtifact, TypeRecord, ValueRecord, VariableRecord,
};
use noir_debugger::foreign_calls::DefaultDebugForeignCallExecutor;
use noirc_printable_type::{PrintableType, PrintableValue, PrintableValueDisplay};

use nargo::NargoError;

#[derive(Clone, Debug)]
struct ActivationFrame {
    function_name: String,
    call_key: usize,
}

pub struct TracingContext<'a, B: BlackBoxFunctionSolver<FieldElement>> {
    debug_context: DebugContext<'a, B>,
    trace_artifact: TraceArtifact,
    last_result: DebugCommandResult,
    previous_line: isize,
    current_line: isize,
    previous_stack_depth: usize,
    current_stack_depth: usize,
    previous_frames: Vec<ActivationFrame>,
    current_frames: Vec<ActivationFrame>,
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
        let last_result = if debug_context.get_current_opcode_location().is_none() {
            // handle circuit with no opcodes
            DebugCommandResult::Done
        } else {
            DebugCommandResult::Ok
        };

        let trace_artifact = TraceArtifact::new();

        Self {
            debug_context,
            trace_artifact,
            last_result,
            previous_line: -1isize,
            current_line: -1isize,
            previous_stack_depth: 0usize,
            current_stack_depth: 0usize,
            previous_frames: Vec::new(),
            current_frames: Vec::new(),
        }
    }

    fn validate_in_progress(&self) -> bool {
        match self.last_result {
            DebugCommandResult::Ok | DebugCommandResult::BreakpointReached(..) => true,
            DebugCommandResult::Done => {
                println!("Execution finished");
                false
            }
            DebugCommandResult::Error(ref error) => {
                println!("ERROR: {}", error);
                false
            }
        }
    }

    fn handle_debug_command_result(&mut self, result: DebugCommandResult) {
        match &result {
            DebugCommandResult::BreakpointReached(location) => {
                println!("Stopped at breakpoint in opcode {}", location);
            }
            DebugCommandResult::Error(error) => {
                println!("ERROR: {}", error);
            }
            _ => (),
        }
        self.last_result = result;
    }

    fn add_top_level_call(&mut self) {
        let call = CallRecord {
            key: 0,
            path_id: 0,
            line: 1337,
            name: String::from("<top-level>"),
            args: Vec::new(),
            return_value: ValueRecord {
                kind: String::from("None"),
                ti: 3,
                elements: None,
                i: None,
                b: None,
                text: None,
            },
            step_id: 0,
            depth: 0,
            parent_key: -1,
        };
        self.trace_artifact.add_call(call);
    }

    fn add_dummy_types(&mut self) {
        self.trace_artifact.add_type(TypeRecord { kind: 7, lang_type: String::from("Fixnum") });
        self.trace_artifact.add_type(TypeRecord { kind: 9, lang_type: String::from("String") });
        self.trace_artifact.add_type(TypeRecord { kind: 0, lang_type: String::from("Array") });
        self.trace_artifact.add_type(TypeRecord { kind: 24, lang_type: String::from("No type") });
    }

    fn next_into(&mut self, num_steps: usize) -> bool {
        self.previous_line = self.current_line;
        self.previous_stack_depth = self.current_stack_depth;
        self.previous_frames = self.current_frames.clone();
        loop {
            if self.validate_in_progress() {
                let result = self.debug_context.next_into();
                let has_more_steps = match result {
                    DebugCommandResult::Done => false,
                    DebugCommandResult::Error(_) => false,
                    _ => true,
                };
                self.handle_debug_command_result(result);

                let call_stack = self.debug_context.get_call_stack();
                let opcode_location = match call_stack.last() {
                    Some(location) => location,
                    None => {
                        println!("stanm: no call stack");
                        return has_more_steps;
                    }
                };

                let frames = self.debug_context.get_variables();
                let current_frame_names =
                    frames.iter().map(|f| String::from(f.function_name)).collect::<Vec<String>>();

                if ((current_frame_names.len() as isize) - (self.previous_frames.len() as isize))
                    > 1
                {
                    todo!("more than one frame entered for a single step\nnew stack: {:?}\n old stack: {:?}", current_frame_names, self.previous_frames);
                }

                let locations =
                    self.debug_context.get_source_location_for_opcode_location(opcode_location);

                let source_location = match locations.last() {
                    Some(location) => location,
                    None => {
                        todo!("stanm: opcode location could not be mapped to source location");
                    }
                };

                let filepath = self.debug_context.get_filepath_for_location(*source_location);
                let path_id = self.trace_artifact.add_or_get_filepath_id(filepath.to_string());
                // get_line_for_location is zero indexed (it seems; no docs)
                self.current_line =
                    self.debug_context.get_line_for_location(*source_location) as isize + 1;

                if self.current_line == self.previous_line {
                    continue;
                } else {
                    // println!("vars: {:?}", self.get_vars_at_last_frame());
                }
                self.current_stack_depth = self.debug_context.get_variables().len();
                let line = self.current_line as usize;

                let call_id = self.trace_artifact.get_next_call_id();
                let step = StepRecord { step_id: num_steps, path_id, line, call_key: call_id };
                self.trace_artifact.add_step(step);
                let vars = self.get_vars_at_last_frame();
                self.trace_artifact.add_vars(vars.iter().map(convert_variable).collect());
                if self.current_stack_depth == self.previous_stack_depth + 1 {
                    let parent_key = if !self.previous_frames.is_empty() {
                        self.previous_frames.last().unwrap().call_key as isize
                    } else {
                        0 // top-level is the parent
                    };
                    let function_name = current_frame_names.last().unwrap().clone();
                    let args = self.get_call_args();
                    let call = CallRecord {
                        key: call_id,
                        path_id,
                        line: 1337, // This doesn't seem to matter to CodeTracer
                        name: function_name.clone(),
                        args,
                        return_value: ValueRecord {
                            kind: String::from("None"),
                            ti: 3,
                            elements: None,
                            i: None,
                            b: None,
                            text: None,
                        },
                        step_id: num_steps,
                        depth: self.current_stack_depth,
                        parent_key,
                    };
                    self.current_frames = self.previous_frames.clone();
                    self.current_frames.push(ActivationFrame { function_name, call_key: call_id });
                    self.trace_artifact.add_call(call);
                } else {
                    for _ in self.current_stack_depth..self.previous_stack_depth {
                        self.current_frames.pop();
                    }
                }

                return has_more_steps;
            } else {
                return false;
            }
        }
    }

    fn get_vars_at_last_frame(&self) -> Vec<(&str, &PrintableValue, &PrintableType)> {
        let frames = self.debug_context.get_variables();
        if frames.is_empty() {
            println!("warning: frames is empty");
            // The first few steps might be before any frames are entered.
            return Vec::new();
        }

        let frame = frames.last().unwrap();

        if frame.variables.is_empty() {
            println!("warning: variables is empty");
        }
        frame.variables.clone()
    }

    fn get_call_args(&mut self) -> Vec<VariableRecord> {
        let mut result = Vec::new();

        let frames = self.debug_context.get_variables();
        if frames.is_empty() {
            println!("warning: frames is empty");
            // The first few steps might be before any frames are entered.
            return Vec::new();
        }

        let frame = frames.last().unwrap();

        if frame.variables.is_empty() {
            println!("warning: variables is empty");
        }

        for var in frame.variables.iter() {
            let var_name = &var.0;
            if frame.function_params.contains(var_name) {
                result.push(convert_variable(var));
            }
        }
        result
    }

    fn is_solved(&self) -> bool {
        self.debug_context.is_solved()
    }

    fn finalize(self) -> WitnessMap<FieldElement> {
        self.debug_context.finalize()
    }
}

fn convert_value(val: &PrintableValue, typ: &PrintableType) -> ValueRecord {
    match typ {
        PrintableType::String { .. } => ValueRecord {
            kind: String::from("String"),
            ti: 1,
            i: None,
            b: None,
            text: match val {
                PrintableValue::String(string_value) => Some(String::from(string_value)),
                _ => todo!("Type-value mismatch: {:?} {:?}", val, typ),
            },
            elements: None,
        },
        PrintableType::Array { length: _, typ: el_type } => ValueRecord {
            kind: String::from("Sequence"),
            ti: 2,
            i: None,
            b: None,
            text: None,
            elements: match val {
                PrintableValue::Vec { array_elements, is_slice: _ } => {
                    Some(array_elements.iter().map(|el| convert_value(el, el_type)).collect())
                }
                _ => todo!("Type-value mismatch: {:?} {:?}", val, typ),
            },
        },
        PrintableType::UnsignedInteger { .. } => ValueRecord {
            kind: String::from("Int"),
            ti: 1,
            i: match val {
                PrintableValue::Field(field_element) => Some(field_element.to_u128() as usize),
                _ => todo!("Type-value mismatch: {:?} {:?}", typ, val),
            },
            b: None,
            text: None,
            elements: None,
        },
        PrintableType::Boolean { .. } => ValueRecord {
            kind: String::from("Bool"),
            ti: 1,
            i: None,
            b: match val {
                PrintableValue::Field(field_element) => Some(field_element.to_u128() != 0),
                _ => todo!("Type-value mismatch: {:?} {:?}", typ, val),
            },
            text: None,
            elements: None,
        },
        _ => todo!("Unsupported type {:?}", typ),
    }
}

fn convert_variable(var: &(&str, &PrintableValue, &PrintableType)) -> VariableRecord {
    (String::from(var.0), convert_value(var.1, var.2))
}

pub fn trace_circuit<B: BlackBoxFunctionSolver<FieldElement>>(
    blackbox_solver: &B,
    circuit: &Circuit<FieldElement>,
    debug_artifact: &DebugArtifact,
    initial_witness: WitnessMap<FieldElement>,
    unconstrained_functions: &[BrilligBytecode<FieldElement>],
) -> Result<TraceArtifact, NargoError> {
    let mut context = TracingContext::new(
        blackbox_solver,
        circuit,
        debug_artifact,
        initial_witness,
        unconstrained_functions,
    );

    let mut num_steps = 0;
    context.add_top_level_call();
    while context.next_into(num_steps) {
        num_steps += 1;
    }
    context.add_dummy_types();
    println!("Tracing steps (len): {:?}", context.trace_artifact.steps.len());

    Ok(context.trace_artifact)
}
