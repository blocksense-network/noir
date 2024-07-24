use crate::{stack_frame::Variable, SourceLocation, StackFrame};

use acvm::acir::AcirField; // necessary, for `to_i128` to work
use acvm::FieldElement;
use noirc_printable_type::{PrintableType, PrintableValue};
use runtime_tracing::{FullValueRecord, Line, Tracer, ValueRecord};
use std::fmt::Write as _;
use std::path::{Path, PathBuf};

/// An object to hold the state needed to implement the glue layer between the Noir tracer and the
/// Tracer in runtime_tracing.
///
/// This is necessary, because there is extra state in this struct, in addition to the mut reference
/// to Tracer.
pub(crate) struct TracerGlue<'a> {
    tracer: &'a mut Tracer,
    // TODO(stanm): add state
}

impl<'a> TracerGlue<'a> {
    pub(crate) fn new(tracer: &'a mut Tracer) -> TracerGlue<'a> {
        TracerGlue { tracer }
    }

    /// Registers a tracing step to the given `location` in the given `tracer`.
    pub(crate) fn register_step(&mut self, location: &SourceLocation) {
        let SourceLocation { filepath, line_number } = &location;
        let path = &PathBuf::from(filepath.to_string());
        let line = Line(*line_number as i64);
        self.tracer.register_step(path, line);
    }

    /// Registers all variables in the given frame for the last registered step. Each time a new step is
    /// registered, all of its variables need to be registered too. If no variables are registered for a
    /// step, the frontend will not carry over the variables registered for the previous step.
    pub(crate) fn register_variables(&mut self, frame: &StackFrame) {
        for variable in &frame.variables {
            self.register_variable(variable);
        }
    }

    /// Registers a variable for the last registered step.
    ///
    /// See `register_variables`.
    fn register_variable(&mut self, variable: &Variable) {
        let value_record = self.register_value(&variable.value, &variable.typ);
        self.tracer.register_variable_with_full_value(&variable.name, value_record);
    }

    /// Registers a value of a given type. Registers the type, if it's the first time it occurs.
    fn register_value(
        &mut self,
        value: &PrintableValue<FieldElement>,
        typ: &PrintableType,
    ) -> ValueRecord {
        match typ {
            PrintableType::Field => {
                if let PrintableValue::Field(field_value) = value {
                    let type_id =
                        self.tracer.ensure_type_id(runtime_tracing::TypeKind::Int, "Field");
                    ValueRecord::Int { i: field_value.to_i128() as i64, type_id }
                } else {
                    // Note(stanm): panic here, because this means the compiler frontend is broken, which
                    // is not the responsibility of this module. Should not be reachable in integration
                    // tests (but reachable in unit tests).
                    //
                    // The same applies for the other `panic!`s in this function.
                    panic!("type-value mismatch: value: {:?} does not match type Field", value)
                }
            }
            PrintableType::UnsignedInteger { width } => {
                if let PrintableValue::Field(field_value) = value {
                    let mut noir_type_name = String::new();
                    if let Err(err) = write!(&mut noir_type_name, "u{width}") {
                        panic!("failed to generate Noir type name: {err}");
                    }
                    let type_id =
                        self.tracer.ensure_type_id(runtime_tracing::TypeKind::Int, &noir_type_name);
                    ValueRecord::Int { i: field_value.to_i128() as i64, type_id }
                } else {
                    panic!(
                        "type-value mismatch: value: {:?} does not match type UnsignedInteger",
                        value
                    )
                }
            }
            _ => {
                // TODO(stanm): cover all types and remove `todo!`.
                todo!("not implemented yet: type that is not Field: {:?}", typ)
            }
        }
    }

    /// Registers a call to the given `frame` at the given `location` in the given `tracer`.
    ///
    /// A helper method, that makes it easier to interface with `Tracer`.
    pub(crate) fn register_call(&mut self, location: &SourceLocation, frame: &StackFrame) {
        let SourceLocation { filepath, line_number } = &location;
        let path = &PathBuf::from(filepath.to_string());
        let line = Line(*line_number as i64);
        let file_id = self.tracer.ensure_function_id(&frame.function_name, path, line);
        let args = self.convert_params_to_args_vec(frame);
        self.tracer.register_call(file_id, args);
    }

    /// Extracts the relevant information from the given `frame` to construct a vector of `ArgRecord`
    /// that the `Tracer` interface expects when registering function calls.
    fn convert_params_to_args_vec(&mut self, frame: &StackFrame) -> Vec<FullValueRecord> {
        let mut result = Vec::new();
        for param_index in &frame.function_param_indexes {
            let variable = &frame.variables[*param_index];
            // TODO(stanm): maybe don't duplicate values?
            let value_record = self.register_value(&variable.value, &variable.typ);
            result.push(self.tracer.arg(&variable.name, value_record));
        }
        result
    }

    /// Register a return statement in the given `tracer`.
    ///
    /// The tracer seems to be keeping context of which function is returning and is not expecting that
    /// to be specified.
    pub(crate) fn register_return(&mut self) {
        let type_id = self.tracer.ensure_type_id(runtime_tracing::TypeKind::None, "()");
        self.tracer.register_return(runtime_tracing::ValueRecord::None { type_id });
    }
}

/// Stores the trace accumulated in `tracer` in the specified directory. The trace is stored as
/// multiple JSON files.
pub fn store_trace(tracer: Tracer, trace_dir: &str) {
    let trace_path = Path::new(trace_dir).join("trace.json");
    match tracer.store_trace_events(&trace_path) {
        Ok(_) => println!("Saved trace to {:?}", trace_path),
        Err(err) => println!("Warning: tracer failed to store trace events: {err}"),
    }

    let trace_path = Path::new(trace_dir).join("trace_metadata.json");
    match tracer.store_trace_metadata(&trace_path) {
        Ok(_) => println!("Saved trace to {:?}", trace_path),
        Err(err) => println!("Warning: tracer failed to store trace metadata: {err}"),
    }

    let trace_path = Path::new(trace_dir).join("trace_paths.json");
    match tracer.store_trace_paths(&trace_path) {
        Ok(_) => println!("Saved trace to {:?}", trace_path),
        Err(err) => println!("Warning: tracer failed to store trace metadata: {err}"),
    }
}
