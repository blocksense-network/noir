use noirc_errors::{CustomDiagnostic, Location, Span};
use nom::{
    Err as NomErr, IResult, Parser,
    error::{ContextError, ErrorKind, ParseError},
};
use std::fmt::Debug;

use super::Input;

// An individual, specific error that occurred.
#[derive(Debug, Clone)]
pub struct ParserError {
    pub span: Span,
    pub kind: ParserErrorKind,
}

// The different kinds of specific errors.
#[derive(Debug, Clone)]
pub enum ParserErrorKind {
    Expected {
        // e.g., "Expected ')'"
        expected: String,
        // e.g., "found 'foo'"
        found: String,
    },
    // e.g., "forall"
    InvalidQuantifier(String),
    // e.g., "reassures(...)"
    UnknownAttribute(String),
    // e.g., "123_abc"
    InvalidIntegerLiteral,
    // e.g., "user.5" where the member access must be an integer literal
    InvalidTupleIndex,
    // When an identifier doesn't follow the rules (e.g., starts with a number)
    InvalidIdentifier(String),
    // A generic message for when other errors don't fit
    Message(String),
}

// The error type that nom's functions will return
#[derive(Debug, Clone)]
pub struct Error {
    pub parser_errors: Vec<ParserError>,
    pub contexts: Vec<String>,
}

// Helper to get a Span from a nom Input
pub fn input_to_span(i: Input) -> Span {
    let offset = i.as_ptr() as usize - i.as_bytes().as_ptr() as usize;
    Span::single_char(offset as u32)
}

/// Builds and returns our custom Error struct directly.
pub fn build_error(input: Input, kind: ParserErrorKind) -> Error {
    Error {
        parser_errors: vec![ParserError { span: input_to_span(input), kind }],
        contexts: vec![],
    }
}

/// From an input slice, get the next token for use in an error message.
pub fn get_found_token(input: Input) -> String {
    match input.split_whitespace().next() {
        Some("") | None => "end of input".to_string(),
        Some(token) => token.to_string(),
    }
}

/// A specialized version of `map_nom_err` for the common "Expected" error.
pub fn expect<'a, P, O>(
    // TODO: maybe `Into<String>` or similar?
    expected_msg: impl AsRef<str>,
    parser: P,
) -> impl FnMut(Input<'a>) -> IResult<Input<'a>, O, Error>
where
    P: Parser<Input<'a>, Output = O, Error = Error>,
{
    map_nom_err(parser, move |fail_input| ParserErrorKind::Expected {
        expected: expected_msg.as_ref().to_string(),
        found: get_found_token(fail_input),
    })
}

/// A combinator that wraps a parser and maps any `NomErr::Error` to a custom error kind.
///
/// It takes a closure `F` that generates the error kind, allowing dynamic error
/// messages based on the input at the failure point.
pub fn map_nom_err<'a, P, O, F>(
    mut parser: P,
    error_fn: F,
) -> impl FnMut(Input<'a>) -> IResult<Input<'a>, O, Error>
where
    P: Parser<Input<'a>, Output = O, Error = Error>,
    F: Fn(Input<'a>) -> ParserErrorKind,
{
    move |input: Input<'a>| {
        parser.parse(input).map_err(|e: NomErr<Error>| {
            if let NomErr::Error(_) = e {
                // Generate the error dynamically using the input at the point of failure.
                NomErr::Error(build_error(input, error_fn(input)))
            } else {
                e
            }
        })
    }
}

impl<'a> ParseError<Input<'a>> for Error {
    /// This function is called by nom's primitives when they fail.
    /// It should create a generic error that can be enriched later by other combinators.
    fn from_error_kind(input: Input<'a>, kind: ErrorKind) -> Self {
        // unreachable!(
        //     "We should wrap all errors and never have to convert from the built-in `nom` ones, still got a {:?} while parsing {:?} tho",
        //     kind, input
        // );
        let err = ParserError {
            span: input_to_span(input),
            // Create a generic message from the nom ErrorKind.
            kind: ParserErrorKind::Message(format!("nom primitive failed: {:?}", kind)),
        };
        Error { parser_errors: vec![err], contexts: vec![] }
    }

    fn append(input: Input<'a>, kind: ErrorKind, mut other: Self) -> Self {
        // // This is called when `alt` fails, for example. We can add to the error stack.
        // let err = ParserError {
        //     span: input_to_span(input),
        //     kind: ParserErrorKind::Message(format!("nom append error: {:?}", kind)),
        // };
        // other.parser_errors.push(err);
        other
    }
}

impl<'a> ContextError<Input<'a>> for Error {
    fn add_context(input: Input<'a>, ctx: &'static str, mut other: Self) -> Self {
        other.contexts.push(ctx.to_string());
        other
    }
}

impl From<ParserError> for CustomDiagnostic {
    fn from(value: ParserError) -> Self {
        // TODO(totel): Get proper location
        let location = Location::dummy();

        let primary_message = match value.kind {
            ParserErrorKind::Expected { expected, found } => {
                format!("Expected {} but found {}", expected, found)
            }
            ParserErrorKind::InvalidQuantifier(q) => format!("Invalid quantifier {}", q),
            ParserErrorKind::UnknownAttribute(attr) => format!("Unknown attribute {}", attr),
            ParserErrorKind::InvalidIntegerLiteral => "Invalid integer literal".to_string(),
            ParserErrorKind::InvalidTupleIndex => "Invalid tuple index".to_string(),
            ParserErrorKind::InvalidIdentifier(identifier) => format!("Invalid identifier {}", identifier),
            ParserErrorKind::Message(msg) => msg,
        };

        CustomDiagnostic::simple_error(primary_message, String::new(), location)
    }
}
