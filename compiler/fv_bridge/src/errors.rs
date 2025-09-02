use fm::FileId;
use formal_verification::{
    parse::errors::{self, ParserError, ParserErrorKind, ParserErrorWithLocation},
    typing::TypeInferenceError,
};
use noirc_driver::CompileError;
use noirc_errors::{CustomDiagnostic, Location};
use noirc_frontend::{
    hir::{resolution::errors::ResolverError, type_check::TypeCheckError},
    monomorphization::{ast::Type, errors::MonomorphizationError},
    parser::ParserError as NoirParserError,
};
use thiserror::Error;

use crate::typed_attrs_to_vir::signed_field_from_bigint_wrapping;

pub(crate) enum MonomorphizationErrorBundle {
    MonomorphizationError(MonomorphizationError),
    FvError(FvMonomorphizationError),
    TypeError(TypeCheckError),
    ParserErrors(Vec<ParserErrorWithLocation>),
    ResolverError(ResolverError),
}

pub(crate) enum CompilationErrorBundle {
    CompileError(CompileError),
    FvError(FvMonomorphizationError),
    TypeError(TypeCheckError),
    ParserErrors(Vec<ParserErrorWithLocation>),
    ResolverError(ResolverError),
}

#[derive(Error, Debug, Clone)]
pub(crate) enum FvMonomorphizationError {
    #[error("Non-ghost function {func_name} was called in FV annotation")]
    ExecInSpecError { func_name: String, location: Location },
}

impl From<FvMonomorphizationError> for CustomDiagnostic {
    fn from(value: FvMonomorphizationError) -> Self {
        match value {
            FvMonomorphizationError::ExecInSpecError { func_name, location } => {
                CustomDiagnostic::simple_error(
                    format!("Non-ghost function {func_name} was called in FV annotation"),
                    String::new(),
                    location,
                )
            }
        }
    }
}

impl From<TypeInferenceError> for MonomorphizationErrorBundle {
    fn from(value: TypeInferenceError) -> Self {
        match value {
            TypeInferenceError::MonomorphizationRequest(monomorphization_request) => {
                panic!("Monomorphization request can not be converted to an error")
            }
            TypeInferenceError::IntegerLiteralDoesNotFit {
                literal,
                literal_type,
                fit_into,
                location,
                message,
            } => MonomorphizationErrorBundle::TypeError(
                TypeCheckError::IntegerLiteralDoesNotFitItsType {
                    expr: signed_field_from_bigint_wrapping(literal),
                    ty: noirc_frontend::Type::Unit, // We present the range which is enough
                    range: {
                        match fit_into {
                            Some(Type::Integer(_, bit_size)) => bit_size.to_string(),
                            _ => "Unknown range".to_string(),
                        }
                    },
                    location,
                },
            ),
            TypeInferenceError::NoirTypeError(type_check_error) => {
                MonomorphizationErrorBundle::TypeError(type_check_error)
            }
        }
    }
}
