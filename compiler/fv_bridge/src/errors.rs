use fm::FileId;
use formal_verification::{
    parse::errors::{self, ParserError, ParserErrorKind},
    typing::TypeInferenceError,
};
use noirc_driver::CompileError;
use noirc_errors::{CustomDiagnostic, Location};
use noirc_frontend::{
    hir::{resolution::errors::ResolverError, type_check::TypeCheckError},
    monomorphization::{ast::Type, errors::MonomorphizationError},
    parser::ParserError as NoirParserError,
};

use crate::typed_attrs_to_vir::signed_field_from_bigint_wrapping;

pub(crate) enum MonomorphizationErrorBundle {
    MonomorphizationError(MonomorphizationError),
    ResolverErrors(Vec<ResolverError>),
    TypeError(TypeCheckError),
    ParserErrors(Vec<ParserError>),
}

pub(crate) enum CompilationErrorBundle {
    CompileError(CompileError),
    ResolverErrors(Vec<ResolverError>),
    TypeError(TypeCheckError),
    ParserErrors(Vec<ParserError>),
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
