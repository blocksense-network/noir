use noirc_errors::{Location, Span};
use noirc_evaluator::vir::vir_gen::{
    Attribute, build_span_no_id,
    expr_to_vir::{expr::function_name_to_vir_fun, types::ast_type_to_vir_type},
};
use noirc_frontend::{
    monomorphization::ast::{Expression, FuncId, Function, GlobalId, Type},
    parser::ParserError as NoirParserError,
};
use nom::Err;
#[allow(unused)]
use nom::{
    IResult, Parser,
    branch::alt,
    bytes::complete::{tag, take_while, take_while1},
    character::complete::{alpha1, alphanumeric0, char, digit1, multispace0},
    combinator::{cut, map, opt, recognize, value},
    error::{ContextError, ParseError, context},
    multi::{fold_many0, separated_list0, separated_list1},
    sequence::{delimited, pair, preceded, separated_pair, terminated},
};
use num_bigint::{BigInt, BigUint, Sign};
use num_traits::ConstZero;
use std::{
    cell::LazyCell,
    collections::{BTreeMap, HashMap},
    fmt::{self, Debug},
    sync::Arc,
};
use vir::ast::{
    AutospecUsage, CallTarget, CallTargetKind, Constant, Expr, ExprX, FunX, Ident, IntRange, Path,
    PathX, SpannedTyped, Typ, TypX, VarIdent, VarIdentDisambiguate,
};

struct State<'a> {
    // span: &'a Span,
    full_length: u32,
    location: Location,
    function: &'a Function,
    global_constants: &'a BTreeMap<GlobalId, (String, Type, Expression)>,
    functions: &'a BTreeMap<FuncId, Function>,
}

type Input<'a> = &'a str;

#[derive(Debug)]
struct ParserError(NoirParserError);

type PResult<'a, T> = IResult<Input<'a>, T, ParserError>;

impl<'a> ParseError<Input<'a>> for ParserError {
    fn from_error_kind(input: Input<'a>, kind: nom::error::ErrorKind) -> Self {
        Self(NoirParserError::empty(
            noirc_frontend::token::Token::DoubleDotEqual,
            Location::dummy(),
        ))
    }

    fn append(input: Input<'a>, kind: nom::error::ErrorKind, other: Self) -> Self {
        other
    }
}

// https://github.com/rust-bakery/nom/blob/main/doc/error_management.md

fn abuild_expr(
    state: &State,
    prev_offset: usize,
    after_offset: usize,
    atypx: Arc<TypX>,
    exprx: ExprX,
) -> Expr {
    let span = build_span_no_id(
        "".to_string(),
        Some(Location {
            span: Span::inclusive(
                state.location.span.start() + state.full_length - prev_offset as u32,
                state.location.span.start() + state.full_length - after_offset as u32,
            ),
            file: state.location.file,
        }),
    );

    SpannedTyped::new(&span, &atypx, exprx)
}

fn build_expr(
    state: &State,
    prev_offset: usize,
    after_offset: usize,
    typx: TypX,
    exprx: ExprX,
) -> Expr {
    let span = build_span_no_id(
        "".to_string(),
        Some(Location {
            span: Span::inclusive(
                state.location.span.start() + state.full_length - prev_offset as u32,
                state.location.span.start() + state.full_length - after_offset as u32,
            ),
            file: state.location.file,
        }),
    );

    SpannedTyped::new(&span, &Arc::new(typx), exprx)
}

// TODO: variable/function rustc ids
// TODO: function signature

pub fn parse_attribute<'a>(
    annotation: &'a str,
    location: Location,
    function: &'a Function,
    global_constants: &'a BTreeMap<GlobalId, (String, Type, Expression)>,
    functions: &'a BTreeMap<FuncId, Function>,
) -> Result<Attribute, ()> {
    let state = State {
        full_length: annotation.len() as u32,
        location,
        function,
        global_constants,
        functions,
    };
    build_span_no_id("".to_string(), todo!());
    todo!()
}

fn parse_bool(input: &str) -> PResult<bool> {
    alt((map(tag("true"), |_| true), map(tag("false"), |_| false))).parse(input)
}

fn sign(input: &str) -> PResult<bool> {
    let (input, opt_sign) = opt(alt((
        //
        value(false, tag(&b"-"[..])),
        value(true, tag(&b"+"[..])),
    )))
    .parse(input)?;
    let sign = opt_sign.unwrap_or(true);

    Ok((input, sign))
}

fn parse_int(input: &str) -> PResult<BigInt> {
    let (input, sign) = sign(input)?;
    let (input, digits) = digit1(input)?;

    let biguint = digits
        .chars()
        .map(|c| c.to_digit(10).expect("`digit1` should return digits"))
        .fold(BigUint::ZERO, |acc, d| acc * 10u8 + d);

    let bigint = BigInt::from_biguint(
        match sign {
            true => Sign::Plus,
            false => Sign::Minus,
        },
        biguint,
    );

    Ok((input, bigint))
}

fn parse_constant<'a>(state: &State, input: &'a str) -> PResult<'a, Expr> {
    let prev_offset = input.len();
    let (input, (typx, exprx)) = alt((
        map(parse_bool, |b| (TypX::Bool, ExprX::Const(Constant::Bool(b)))),
        map(parse_int, |bi| {
            // TODO: Better type than `TypX::Int`
            (TypX::Int(IntRange::Int), ExprX::Const(Constant::Int(bi)))
        }),
    ))
    .parse(input)?;

    let after_offset = input.len();

    let res = build_expr(state, prev_offset, after_offset, typx, exprx);

    Ok((input, res))
}

fn parse_identifier(input: &str) -> PResult<&str> {
    fn is_valid_start(c: char) -> bool {
        c.is_ascii_alphabetic() || c == '_'
    }

    fn is_valid_char(c: char) -> bool {
        c.is_ascii_alphanumeric() || c == '_'
    }

    let mut parser = recognize(pair(
        //
        take_while1(is_valid_start),
        take_while(is_valid_char),
    ));

    parser.parse(input)
}

fn parse_var<'a>(state: &State, input: &'a str) -> PResult<'a, Expr> {
    let prev_offset = input.len();
    let (input, ident) = parse_identifier(input)?;
    let after_offset = input.len();

    Ok((
        input,
        abuild_expr(
            state,
            prev_offset,
            after_offset,
            // NOTE: type of variable is either found in the function's parameters
            //       or is the magic `result`, which inherits the function's return type
            state
                .function
                .parameters
                .iter()
                .find_map(|k| (k.2 == ident).then(|| &k.3))
                .or_else(|| (ident == "result").then(|| &state.function.return_type))
                .map(ast_type_to_vir_type)
                .ok_or(Err::Error(ParserError(NoirParserError::empty(
                    noirc_frontend::token::Token::Arrow,
                    Location::dummy(),
                ))))?,
            ExprX::Var(VarIdent(Arc::new(ident.to_string()), VarIdentDisambiguate::RustcId(0))),
        ),
    ))
}

fn parse_fn_call<'a>(state: &State, input: &'a str) -> PResult<'a, Expr> {
    let prev_offset = input.len();
    let (input, name) = parse_identifier(input)?;

    let (input, params) = delimited(
        tag("("),
        separated_list0(pair(tag(","), opt(tag(" "))), |input| parse_expression(state, input)),
        tag(")"),
    )
    .parse(input)?;
    let after_offset = input.len();

    Ok((
        input,
        abuild_expr(
            state,
            prev_offset,
            after_offset,
            state
                .functions
                .iter()
                .find_map(|(_, func)| (func.name == name).then_some(&func.return_type))
                .map(ast_type_to_vir_type)
                .ok_or(Err::Error(ParserError(NoirParserError::empty(
                    noirc_frontend::token::Token::Arrow,
                    Location::dummy(),
                ))))?,
            ExprX::Call(
                CallTarget::Fun(
                    CallTargetKind::Static,
                    function_name_to_vir_fun(name.to_string()),
                    Arc::new(vec![]),
                    Arc::new(vec![]),
                    AutospecUsage::Final,
                ),
                Arc::new(params),
            ),
        ),
    ))
}

fn parse_expression<'a>(state: &State, input: &'a str) -> PResult<'a, Expr> {
    alt((
        //
        |input| parse_fn_call(state, input),
        |input| parse_constant(state, input),
        |input| parse_var(state, input),
    ))
    .parse(input)
}

#[cfg(test)]
mod tests {
    use noirc_frontend::{
        monomorphization::ast::{FuncId, InlineType, LocalId},
        shared::Visibility,
    };

    use super::*;

    fn empty_state(full_length: u32) -> State<'static> {
        State {
            full_length,
            location: Location { span: Span::inclusive(1234, 5678), file: Default::default() },
            function: Box::leak(Box::new(Function {
                id: FuncId(4321),
                name: "tutmanik".to_string(),
                parameters: vec![
                    (LocalId(0), false, "a".to_string(), Type::Bool, Visibility::Public),
                    (LocalId(1), false, "kek".to_string(), Type::Unit, Visibility::Public),
                ],
                body: Expression::Block(vec![]),
                return_type: Type::Unit,
                return_visibility: Visibility::Public,
                unconstrained: false,
                inline_type: InlineType::Inline,
                func_sig: (vec![], None),
                formal_verification_attributes: vec![],
            })),
            global_constants: Box::leak(Box::new(vec![].into_iter().collect())),
            functions: Box::leak(Box::new(
                vec![(
                    FuncId(0),
                    Function {
                        id: FuncId(0),
                        name: "banica".to_string(),
                        // TODO: not type-checking parameters, yet
                        //       might need to do some manual dispatching
                        parameters: vec![],
                        body: Expression::Block(vec![]),
                        return_type: Type::Field,
                        return_visibility: Visibility::Public,
                        unconstrained: false,
                        inline_type: InlineType::Inline,
                        func_sig: (vec![], None),
                        formal_verification_attributes: vec![],
                    },
                )]
                .into_iter()
                .collect(),
            )),
        }
    }

    #[test]
    fn test_bool_true() {
        let expr = parse_expression(&empty_state(4), "true").unwrap();
        assert_eq!(expr.0, "");
        assert!(matches!(*expr.1.typ, TypX::Bool));
        assert!(matches!(expr.1.x, ExprX::Const(Constant::Bool(true))));
    }

    #[test]
    fn test_int() {
        let chislo = "1234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890";
        let expr = parse_expression(&empty_state(100), chislo).unwrap();
        assert_eq!(expr.0, "");
        assert!(matches!(*expr.1.typ, TypX::Int(IntRange::Int)));
        let ExprX::Const(Constant::Int(ref bi)) = expr.1.x else { panic!() };
        assert_eq!(bi.to_str_radix(10), chislo);
    }

    #[test]
    fn test_ident() {
        let identche = "Banica_123_";
        let expr = parse_expression(&empty_state(11), identche).unwrap();
        assert_eq!(expr.0, "");
        // TODO: same as definition
        assert!(matches!(*expr.1.typ, TypX::Bool));
        let ExprX::Var(VarIdent(ref i, _)) = expr.1.x else { panic!() };
        assert_eq!(**i, identche.to_string());
    }

    #[test]
    #[should_panic]
    fn test_ident_starts_with_digit() {
        let identche = "1Banica_123_";
        let expr = parse_var(&empty_state(12), identche).unwrap();
        assert_eq!(expr.0, "");
        dbg!(expr);
    }

    #[test]
    fn test_function_call() {
        let identche = "banica(1, banica(a, kek))";
        let expr = parse_expression(&empty_state(25), identche).unwrap();
        assert_eq!(expr.0, "");
        dbg!(expr);
    }
}
