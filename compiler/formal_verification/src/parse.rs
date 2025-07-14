use noirc_errors::{Location, Span};
use nom::{
    Err, IResult, Parser,
    branch::alt,
    bytes::complete::{tag, take_while, take_while1},
    character::complete::{digit1 as digit, multispace0 as multispace},
    combinator::{map, opt, recognize, value},
    error::{ErrorKind, ParseError},
    multi::{many0, separated_list0},
    sequence::{delimited, pair, preceded, terminated},
};
use num_bigint::{BigInt, BigUint, Sign};
use std::fmt::Debug;

use crate::ast::{
    ArithmeticOp, BinaryOp, BooleanOp, ExprF, Literal, OffsetExpr, PredicateOp, UnaryOp,
};

#[derive(Debug)]
pub enum ParserError {
    // TODO: real errors
    Oops,
}

#[derive(Debug)]
pub struct Error {
    pub parser_errors: Vec<ParserError>,
}

pub type Input<'a> = &'a str;
pub type PResult<'a, T> = IResult<Input<'a>, T, Error>;

impl<'a, 'b> ParseError<Input<'a>> for Error {
    fn from_error_kind(input: Input<'a>, kind: ErrorKind) -> Self {
        Self {
            parser_errors: vec![ParserError::Oops],
            // location: build_location(input.len(), input.len()),
        }
    }

    fn append(input: Input<'a>, kind: ErrorKind, mut other: Self) -> Self {
        // TODO: smart stuff
        other.parser_errors.push(ParserError::Oops);
        other
    }
}

// https://github.com/rust-bakery/nom/blob/main/doc/error_management.md

pub(crate) fn build_location(
    annotation_location: Location,
    full_length: u32,
    prev_offset: u32,
    after_offset: u32,
) -> Location {
    Location {
        span: Span::inclusive(
            annotation_location.span.start() + full_length - prev_offset,
            annotation_location.span.start() + full_length - after_offset,
        ),
        file: annotation_location.file,
    }
}

pub(crate) fn build_expr(
    prev_offset: usize,
    after_offset: usize,
    exprf: ExprF<OffsetExpr>,
) -> OffsetExpr {
    OffsetExpr { ann: (prev_offset as u32, after_offset as u32), expr: Box::new(exprf) }
}

pub(crate) fn build_offset_from_exprs(left: &OffsetExpr, right: &OffsetExpr) -> (u32, u32) {
    (left.ann.0, right.ann.1)
}

// TODO: array indexing - ast_index_to_vir_expr
// TODO: tuple indexing - ast_tuple_access_to_vir_expr

pub(crate) fn parse_bool<'a, 'b>(input: Input<'a>) -> PResult<'a, bool> {
    alt((map(tag("true"), |_| true), map(tag("false"), |_| false))).parse(input)
}

pub(crate) fn parse_sign<'a, 'b>(input: Input<'a>) -> PResult<'a, bool> {
    let (input, opt_sign) = opt(alt((
        //
        value(false, tag(&b"-"[..])),
        value(true, tag(&b"+"[..])),
    )))
    .parse(input)?;
    let sign = opt_sign.unwrap_or(true);

    Ok((input, sign))
}

pub(crate) fn parse_int<'a, 'b>(input: Input<'a>) -> PResult<'a, BigInt> {
    let (input, sign) = parse_sign(input)?;
    let (input, digits) = digit(input)?;

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

pub(crate) fn parse_constant_expr<'a, 'b>(input: Input<'a>) -> PResult<'a, OffsetExpr> {
    // eprintln!("trying constant: {}", input);
    let prev_offset = input.len();
    let (input, exprf) = alt((
        map(parse_bool, |b| ExprF::Literal { value: Literal::Bool(b) }),
        map(parse_int, |bi| ExprF::Literal { value: Literal::Int(bi) }),
    ))
    .parse(input)?;

    let after_offset = input.len();

    let res = build_expr(prev_offset, after_offset, exprf);

    Ok((input, res))
}

// TODO: parse identifier expression
// TODO: parse module references `fv_std::SOMETHING`
pub(crate) fn parse_identifier<'a, 'b>(input: Input<'a>) -> PResult<'a, &'a str> {
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

    let (input, name) = parser.parse(input)?;

    Ok((input, name))
}

pub(crate) fn parse_var_expr<'a, 'b>(input: Input<'a>) -> PResult<'a, OffsetExpr> {
    // eprintln!("trying var: {}", input);
    let prev_offset = input.len();
    let (input, ident) = parse_identifier(input).map_err(|_| {
        // Err::Error(Error {
        //     resolver_errors: vec![ResolverError::ParserError(Box::new(
        //         NoirParserError::with_reason(
        //             // TODO: ne
        //             ParserErrorReason::DocCommentDoesNotDocumentAnything,
        //             Location::dummy(),
        //         ),
        //     ))],
        //     location: Location::dummy(),
        // })
        Err::Error(Error { parser_errors: vec![ParserError::Oops] })
    })?;
    let after_offset = input.len();

    Ok((input, build_expr(prev_offset, after_offset, ExprF::Variable { name: ident.to_string() })))
}

pub(crate) fn parse_fn_call_expr<'a, 'b>(input: Input<'a>) -> PResult<'a, OffsetExpr> {
    // eprintln!("trying call: {}", input);
    let prev_offset = input.len();
    let (input, name) = parse_identifier(input)?;

    let (input, params) = delimited(
        tag("("),
        separated_list0(pair(tag(","), multispace), parse_expression_expr),
        tag(")"),
    )
    .parse(input)?;
    let after_offset = input.len();

    Ok((
        input,
        build_expr(
            prev_offset,
            after_offset,
            ExprF::FnCall { name: name.to_string(), args: params },
        ),
    ))
}

pub(crate) fn parse_additive_expr<'a, 'b>(input: Input<'a>) -> PResult<'a, OffsetExpr> {
    // eprintln!("trying additive: {}", input);
    let (input, (first, remainder)) = pair(
        parse_multiplicative_expr,
        many0(pair(
            delimited(multispace, alt((tag("+"), tag("-"))), multispace),
            parse_multiplicative_expr,
        )),
    )
    .parse(input)?;

    Ok((
        input,
        remainder.into_iter().fold(first, |expr_left, (op, expr_right)| {
            let op_kind = match op {
                "+" => BinaryOp::ArithmeticOp(ArithmeticOp::Add),
                "-" => BinaryOp::ArithmeticOp(ArithmeticOp::Sub),
                _ => unreachable!(),
            };
            OffsetExpr {
                ann: build_offset_from_exprs(&expr_left, &expr_right),
                expr: Box::new(ExprF::BinaryOp { op: op_kind, expr_left, expr_right }),
            }
        }),
    ))
}

pub(crate) fn parse_multiplicative_expr<'a, 'b>(input: Input<'a>) -> PResult<'a, OffsetExpr> {
    // eprintln!("trying multiplicative: {}", input);
    let (input, (first, remainder)) = pair(
        parse_unary_expr,
        many0(pair(
            delimited(multispace, alt((tag("*"), tag("/"), tag("%"))), multispace),
            parse_unary_expr,
        )),
    )
    .parse(input)?;

    Ok((
        input,
        remainder.into_iter().fold(first, |expr_left, (op, expr_right)| {
            let op_kind = match op {
                "*" => BinaryOp::ArithmeticOp(ArithmeticOp::Mul),
                "/" => BinaryOp::ArithmeticOp(ArithmeticOp::Div),
                "%" => BinaryOp::ArithmeticOp(ArithmeticOp::Mod),
                _ => unreachable!(),
            };
            OffsetExpr {
                ann: build_offset_from_exprs(&expr_left, &expr_right),
                expr: Box::new(ExprF::BinaryOp { op: op_kind, expr_left, expr_right }),
            }
        }),
    ))
}

pub(crate) fn parse_expression_expr<'a, 'b>(input: Input<'a>) -> PResult<'a, OffsetExpr> {
    // eprintln!("trying expression: {}", input);
    // NOTE: Start parsing from the lowest precedence operator
    alt((
        //
        parse_implication_expr,
    ))
    .parse(input)
}

pub(crate) fn parse_parenthesised_expr<'a, 'b>(input: Input<'a>) -> PResult<'a, OffsetExpr> {
    // eprintln!("trying parenthesised: {}", input);
    delimited(multispace, delimited(tag("("), parse_expression_expr, tag(")")), multispace)
        .parse(input)
}

pub(crate) fn parse_primary_expr<'a, 'b>(input: Input<'a>) -> PResult<'a, OffsetExpr> {
    // eprintln!("trying primary: {}", input);
    alt((
        //
        parse_parenthesised_expr,
        parse_fn_call_expr,
        parse_constant_expr,
        parse_var_expr,
    ))
    .parse(input)
}

pub(crate) fn parse_unary_expr<'a, 'b>(input: Input<'a>) -> PResult<'a, OffsetExpr> {
    // eprintln!("trying unary: {}", input);
    alt((
        map(preceded(terminated(tag("!"), multispace), parse_unary_expr), |expr| OffsetExpr {
            ann: expr.ann,
            expr: Box::new(ExprF::UnaryOp { op: UnaryOp::Not, expr }),
        }),
        parse_primary_expr,
    ))
    .parse(input)
}

pub(crate) fn parse_comparison_expr<'a, 'b>(input: Input<'a>) -> PResult<'a, OffsetExpr> {
    // eprintln!("trying comparison: {}", input);
    let (input, mut expr_left) = parse_additive_expr(input)?;

    // Comparison operators are not associative (e.g., `a < b < c` is invalid),
    // so we just look for one optional occurrence.
    if let Ok((input, (op, expr_right))) = pair(
        delimited(
            multispace,
            alt((tag("=="), tag("!="), tag("<="), tag(">="), tag("<"), tag(">"))),
            multispace,
        ),
        parse_additive_expr,
    )
    .parse(input)
    {
        let op_kind = match op {
            "==" => BinaryOp::PredicateOp(PredicateOp::Eq),
            "!=" => BinaryOp::PredicateOp(PredicateOp::Neq),
            "<" => BinaryOp::PredicateOp(PredicateOp::Lt),
            "<=" => BinaryOp::PredicateOp(PredicateOp::Le),
            ">" => BinaryOp::PredicateOp(PredicateOp::Gt),
            ">=" => BinaryOp::PredicateOp(PredicateOp::Ge),
            _ => unreachable!(),
        };
        expr_left = OffsetExpr {
            ann: build_offset_from_exprs(&expr_left, &expr_right),
            expr: Box::new(ExprF::BinaryOp { op: op_kind, expr_left, expr_right }),
        };

        return Ok((input, expr_left));
    }

    Ok((input, expr_left))
}

pub(crate) fn parse_logical_and_expr<'a, 'b>(input: Input<'a>) -> PResult<'a, OffsetExpr> {
    // eprintln!("trying and: {}", input);
    let (input, (first, remainder)) = pair(
        parse_comparison_expr,
        many0(pair(delimited(multispace, tag("&"), multispace), parse_comparison_expr)),
    )
    .parse(input)?;

    Ok((
        input,
        remainder.into_iter().fold(first, |expr_left, (_op, expr_right)| OffsetExpr {
            ann: build_offset_from_exprs(&expr_left, &expr_right),
            expr: Box::new(ExprF::BinaryOp {
                op: BinaryOp::BooleanOp(BooleanOp::And),
                expr_left,
                expr_right,
            }),
        }),
    ))
}

pub(crate) fn parse_logical_or_expr<'a, 'b>(input: Input<'a>) -> PResult<'a, OffsetExpr> {
    // eprintln!("trying or: {}", input);
    let (input, (first, remainder)) = pair(
        parse_logical_and_expr,
        many0(pair(delimited(multispace, tag("|"), multispace), parse_logical_and_expr)),
    )
    .parse(input)?;

    Ok((
        input,
        remainder.into_iter().fold(first, |expr_left, (_op, expr_right)| OffsetExpr {
            ann: build_offset_from_exprs(&expr_left, &expr_right),
            expr: Box::new(ExprF::BinaryOp {
                op: BinaryOp::BooleanOp(BooleanOp::Or),
                expr_left,
                expr_right,
            }),
        }),
    ))
}

pub(crate) fn parse_implication_expr<'a, 'b>(input: Input<'a>) -> PResult<'a, OffsetExpr> {
    // eprintln!("trying implication: {}", input);
    let (input, (first, remainder)) = pair(
        parse_logical_or_expr,
        many0(pair(delimited(multispace, tag("==>"), multispace), parse_logical_or_expr)),
    )
    .parse(input)?;

    Ok((
        input,
        remainder.into_iter().fold(first, |expr_left, (_op, expr_right)| OffsetExpr {
            ann: build_offset_from_exprs(&expr_left, &expr_right),
            expr: Box::new(ExprF::BinaryOp {
                op: BinaryOp::BooleanOp(BooleanOp::Implies),
                expr_left,
                expr_right,
            }),
        }),
    ))
}

#[cfg(test)]
pub mod tests {
    use noirc_frontend::{
        monomorphization::ast::{
            Expression, FuncId, Function, InlineType, LocalId, Type as NoirType,
        },
        shared::Visibility,
    };

    use crate::{Attribute, State, ast::Literal, parse_attribute};

    use super::*;

    pub fn empty_state(full_length: u32) -> State<'static> {
        State {
            full_length,
            location: Location { span: Span::inclusive(1000, 2000), file: Default::default() },
            function: Box::leak(Box::new(Function {
                id: FuncId(4321),
                name: "tutmanik".to_string(),
                parameters: vec![
                    (
                        LocalId(0),
                        false,
                        "a".to_string(),
                        NoirType::Integer(
                            noirc_frontend::shared::Signedness::Signed,
                            noirc_frontend::ast::IntegerBitSize::ThirtyTwo,
                        ),
                        Visibility::Public,
                    ),
                    (LocalId(1), false, "kek".to_string(), NoirType::Unit, Visibility::Public),
                    (
                        LocalId(2),
                        false,
                        "Banica_123_".to_string(),
                        NoirType::Bool,
                        Visibility::Public,
                    ),
                ],
                body: Expression::Block(vec![]),
                return_type: NoirType::Integer(
                    noirc_frontend::shared::Signedness::Signed,
                    noirc_frontend::ast::IntegerBitSize::ThirtyTwo,
                ),
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
                        return_type: NoirType::Field,
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

    pub fn parse<'a, 'b>(input: &'a str) -> PResult<'a, OffsetExpr> {
        parse_expression_expr(input)
    }

    #[test]
    fn test_bool_true() {
        let (input, expr) = parse("true").unwrap();
        assert_eq!(input, "");
        // assert!(matches!(*expr.1.typ, TypX::Bool));
        assert!(matches!(*expr.expr, ExprF::Literal { value: Literal::Bool(true) }));
    }

    #[test]
    fn test_int() {
        let chislo = "1234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890";
        let (input, expr) = parse(chislo).unwrap();
        assert_eq!(input, "");
        // assert!(matches!(*expr.1.typ, TypX::Int(IntRange::Int)));
        let ExprF::Literal { value: Literal::Int(ref bi) } = *expr.expr else { panic!() };
        assert_eq!(bi.to_str_radix(10), chislo);
    }

    #[test]
    fn test_ident() {
        let identche = "Banica_123_";
        let (input, expr) = parse(identche).unwrap();
        assert_eq!(input, "");
        // assert!(matches!(*expr.1.typ, TypX::Bool));
        let ExprF::Variable { name: i } = *expr.expr else { panic!() };
        assert_eq!(&i, identche);
    }

    #[test]
    #[should_panic]
    fn test_ident_starts_with_digit() {
        let identche = "1Banica_123_";
        let expr = parse_var_expr(identche).unwrap();
        assert_eq!(expr.0, "");
        dbg!(expr);
    }

    #[test]
    fn test_function_call() {
        let expr = parse("banica(1, banica(a, kek))").unwrap();
        assert_eq!(expr.0, "");
        dbg!(expr);
    }

    #[test]
    fn test_sum() {
        let identche = "1 + 2 * 3";
        let expr = parse(identche).unwrap();
        assert_eq!(expr.0, "");
        dbg!(expr);
    }

    #[test]
    fn test_ghost() {
        let annotation = "ghost";
        let state = empty_state(annotation.len() as u32);
        let attribute = parse_attribute(
            annotation,
            state.location,
            state.function,
            state.global_constants,
            state.functions,
        )
        .unwrap();

        assert!(matches!(attribute, Attribute::Ghost));
    }
}
