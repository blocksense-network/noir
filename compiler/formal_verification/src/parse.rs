use noirc_errors::{Location, Span};
use nom::{
    Err, IResult, Parser,
    branch::alt,
    bytes::complete::{tag, take_while, take_while1},
    character::complete::{digit1 as digit, multispace0 as multispace},
    combinator::{cut, fail, map, opt, recognize, value},
    error::{ContextError, ErrorKind, ParseError, context},
    multi::{many0, separated_list0},
    sequence::{delimited, pair, preceded, terminated},
};
use num_bigint::{BigInt, BigUint, Sign};
use std::fmt::Debug;

use crate::ast::{BinaryOp, ExprF, Identifier, Literal, OffsetExpr, Quantifier, UnaryOp};

#[derive(Debug)]
pub enum ParserError {
    // TODO: real errors
    Oops,
}

#[derive(Debug)]
pub struct Error {
    pub parser_errors: Vec<ParserError>,
    pub contexts: Vec<String>,
}

pub type Input<'a> = &'a str;
pub type PResult<'a, T> = IResult<Input<'a>, T, Error>;

impl<'a> ParseError<Input<'a>> for Error {
    fn from_error_kind(input: Input<'a>, kind: ErrorKind) -> Self {
        Self {
            parser_errors: vec![ParserError::Oops],
            contexts: vec![],
            // location: build_location(input.len(), input.len()),
        }
    }

    fn append(input: Input<'a>, kind: ErrorKind, mut other: Self) -> Self {
        // TODO: smart stuff
        other.parser_errors.push(ParserError::Oops);
        other
    }
}

impl<'a> ContextError<Input<'a>> for Error {
    fn add_context(input: Input<'a>, ctx: &'static str, mut other: Self) -> Self {
        other.contexts.push(ctx.to_string());
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

/*************************************************************************
* Main parser combinators, in order of precedence, like in upstream Noir *
**************************************************************************/

// TODO: array indexing - ast_index_to_vir_expr
// TODO: tuple indexing - ast_tuple_access_to_vir_expr

pub(crate) fn parse_expression<'a>(input: Input<'a>) -> PResult<'a, OffsetExpr> {
    alt((
        //
        parse_implication_expr,
    ))
    .parse(input)
}

pub(crate) fn parse_implication_expr<'a>(input: Input<'a>) -> PResult<'a, OffsetExpr> {
    let (input, (first, remainder)) = pair(
        parse_equality_expr,
        context(
            "implication",
            many0(pair(delimited(multispace, tag("==>"), multispace), parse_equality_expr)),
        ),
    )
    .parse(input)?;

    Ok((
        input,
        remainder.into_iter().fold(first, |expr_left, (op, expr_right)| {
            let op_kind = match op {
                "==>" => BinaryOp::Implies,
                _ => unreachable!(),
            };
            OffsetExpr {
                ann: build_offset_from_exprs(&expr_left, &expr_right),
                expr: Box::new(ExprF::BinaryOp { op: op_kind, expr_left, expr_right }),
            }
        }),
    ))
}

pub(crate) fn parse_equality_expr<'a>(input: Input<'a>) -> PResult<'a, OffsetExpr> {
    let (input, (first, remainder)) = pair(
        parse_or_expr,
        context(
            "equality",
            many0(pair(
                delimited(multispace, alt((tag("=="), tag("!="))), multispace),
                parse_or_expr,
            )),
        ),
    )
    .parse(input)?;

    Ok((
        input,
        remainder.into_iter().fold(first, |expr_left, (op, expr_right)| {
            let op_kind = match op {
                "==" => BinaryOp::Eq,
                "!=" => BinaryOp::Neq,
                _ => unreachable!(),
            };
            OffsetExpr {
                ann: build_offset_from_exprs(&expr_left, &expr_right),
                expr: Box::new(ExprF::BinaryOp { op: op_kind, expr_left, expr_right }),
            }
        }),
    ))
}

pub(crate) fn parse_or_expr<'a>(input: Input<'a>) -> PResult<'a, OffsetExpr> {
    let (input, (first, remainder)) = pair(
        parse_and_expr,
        context("or", many0(pair(delimited(multispace, tag("|"), multispace), parse_and_expr))),
    )
    .parse(input)?;

    Ok((
        input,
        remainder.into_iter().fold(first, |expr_left, (op, expr_right)| {
            let op_kind = match op {
                "|" => BinaryOp::Or,
                _ => unreachable!(),
            };
            OffsetExpr {
                ann: build_offset_from_exprs(&expr_left, &expr_right),
                expr: Box::new(ExprF::BinaryOp { op: op_kind, expr_left, expr_right }),
            }
        }),
    ))
}

pub(crate) fn parse_and_expr<'a>(input: Input<'a>) -> PResult<'a, OffsetExpr> {
    let (input, (first, remainder)) = pair(
        parse_xor_expr,
        context("and", many0(pair(delimited(multispace, tag("&"), multispace), parse_xor_expr))),
    )
    .parse(input)?;

    Ok((
        input,
        remainder.into_iter().fold(first, |expr_left, (op, expr_right)| {
            let op_kind = match op {
                "&" => BinaryOp::And,
                _ => unreachable!(),
            };
            OffsetExpr {
                ann: build_offset_from_exprs(&expr_left, &expr_right),
                expr: Box::new(ExprF::BinaryOp { op: op_kind, expr_left, expr_right }),
            }
        }),
    ))
}

pub(crate) fn parse_xor_expr<'a>(input: Input<'a>) -> PResult<'a, OffsetExpr> {
    let (input, (first, remainder)) = pair(
        parse_comparison_expr,
        context(
            "xor",
            many0(pair(delimited(multispace, tag("^"), multispace), parse_comparison_expr)),
        ),
    )
    .parse(input)?;

    Ok((
        input,
        remainder.into_iter().fold(first, |expr_left, (op, expr_right)| {
            let op_kind = match op {
                "^" => BinaryOp::Xor,
                _ => unreachable!(),
            };
            OffsetExpr {
                ann: build_offset_from_exprs(&expr_left, &expr_right),
                expr: Box::new(ExprF::BinaryOp { op: op_kind, expr_left, expr_right }),
            }
        }),
    ))
}
pub(crate) fn parse_comparison_expr<'a>(input: Input<'a>) -> PResult<'a, OffsetExpr> {
    let (input, mut expr_left) = parse_shift_expr(input)?;

    // Comparison operators are not associative (e.g., `a < b < c` is invalid),
    // so we just look for one optional occurrence.
    if let Ok((input, (op, expr_right))) = pair(
        context(
            "comparison",
            delimited(multispace, alt((tag("<="), tag(">="), tag("<"), tag(">"))), multispace),
        ),
        parse_shift_expr,
    )
    .parse(input)
    {
        let op_kind = match op {
            "<" => BinaryOp::Lt,
            "<=" => BinaryOp::Le,
            ">" => BinaryOp::Gt,
            ">=" => BinaryOp::Ge,
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

pub(crate) fn parse_shift_expr<'a>(input: Input<'a>) -> PResult<'a, OffsetExpr> {
    let (input, mut expr_left) = parse_additive_expr(input)?;

    // Comparison operators are not associative (e.g., `a < b < c` is invalid),
    // so we just look for one optional occurrence.
    if let Ok((input, (op, expr_right))) = pair(
        context(
            "comparison",
            delimited(multispace, alt((tag("<="), tag(">="), tag("<"), tag(">"))), multispace),
        ),
        parse_additive_expr,
    )
    .parse(input)
    {
        let op_kind = match op {
            "<" => BinaryOp::Lt,
            "<=" => BinaryOp::Le,
            ">" => BinaryOp::Gt,
            ">=" => BinaryOp::Ge,
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

pub(crate) fn parse_additive_expr<'a>(input: Input<'a>) -> PResult<'a, OffsetExpr> {
    let (input, (first, remainder)) = pair(
        parse_multiplicative_expr,
        context(
            "additive",
            many0(pair(
                delimited(multispace, alt((tag("+"), tag("-"))), multispace),
                parse_multiplicative_expr,
            )),
        ),
    )
    .parse(input)?;

    Ok((
        input,
        remainder.into_iter().fold(first, |expr_left, (op, expr_right)| {
            let op_kind = match op {
                "+" => BinaryOp::Add,
                "-" => BinaryOp::Sub,
                _ => unreachable!(),
            };
            OffsetExpr {
                ann: build_offset_from_exprs(&expr_left, &expr_right),
                expr: Box::new(ExprF::BinaryOp { op: op_kind, expr_left, expr_right }),
            }
        }),
    ))
}

pub(crate) fn parse_multiplicative_expr<'a>(input: Input<'a>) -> PResult<'a, OffsetExpr> {
    let (input, (first, remainder)) = pair(
        parse_prefix_expr,
        context(
            "multiplicative",
            many0(pair(
                delimited(multispace, alt((tag("*"), tag("/"), tag("%"))), multispace),
                parse_prefix_expr,
            )),
        ),
    )
    .parse(input)?;

    Ok((
        input,
        remainder.into_iter().fold(first, |expr_left, (op, expr_right)| {
            let op_kind = match op {
                "*" => BinaryOp::Mul,
                "/" => BinaryOp::Div,
                "%" => BinaryOp::Mod,
                _ => unreachable!(),
            };
            OffsetExpr {
                ann: build_offset_from_exprs(&expr_left, &expr_right),
                expr: Box::new(ExprF::BinaryOp { op: op_kind, expr_left, expr_right }),
            }
        }),
    ))
}

pub(crate) fn parse_prefix_expr<'a>(input: Input<'a>) -> PResult<'a, OffsetExpr> {
    alt((
        context(
            "prefix",
            map(preceded(terminated(tag("!"), multispace), parse_prefix_expr), |expr| OffsetExpr {
                ann: expr.ann,
                expr: Box::new(ExprF::UnaryOp { op: UnaryOp::Not, expr }),
            }),
        ),
        parse_postfix_expr,
    ))
    .parse(input)
}

enum Postfix {
    ArrayIndex(OffsetExpr),
    TupleMember(BigInt),
}

pub(crate) fn parse_postfix_expr<'a>(input: Input<'a>) -> PResult<'a, OffsetExpr> {
    let (input, base_expr) = parse_atom_expr(input)?;

    let (input, suffixes) = many0(parse_any_suffix).parse(input)?;

    let final_expr = suffixes.into_iter().try_fold(base_expr, |current_expr, suffix| {
        // TODO: real span
        let new_ann = current_expr.ann;

        Ok(match suffix {
            Postfix::ArrayIndex(index_expr) => {
                let ann = build_offset_from_exprs(&current_expr, &index_expr);
                OffsetExpr {
                    ann,
                    expr: Box::new(ExprF::Index { expr: current_expr, index: index_expr }),
                }
            }
            Postfix::TupleMember(index) => OffsetExpr {
                ann: new_ann,
                expr: Box::new(ExprF::TupleAccess {
                    expr: current_expr,
                    index: index.try_into().map_err(|_| {
                        nom::Err::Error(Error { parser_errors: vec![], contexts: vec![] })
                    })?,
                }),
            },
        })
    })?;

    Ok((input, final_expr))
}

fn parse_any_suffix<'a>(input: Input<'a>) -> PResult<'a, Postfix> {
    alt((
        map(parse_index_suffix, Postfix::ArrayIndex),
        map(parse_member_suffix, Postfix::TupleMember),
    ))
    .parse(input)
}

/// Parses an index suffix `[expr]` and returns just the inner expression.
fn parse_index_suffix<'a>(input: Input<'a>) -> PResult<'a, OffsetExpr> {
    preceded(
        multispace,
        delimited(tag("["), cut(delimited(multispace, parse_expression, multispace)), tag("]")),
    )
    .parse(input)
}

/// Parses a member access `.field` and returns just the field identifier.
fn parse_member_suffix<'a>(input: Input<'a>) -> PResult<'a, BigInt> {
    preceded(pair(multispace, tag(".")), cut(parse_int)).parse(input)
}

pub(crate) fn parse_atom_expr<'a>(input: Input<'a>) -> PResult<'a, OffsetExpr> {
    alt((
        context("parenthesised", parse_parenthesised_expr),
        context("quantifier", parse_quantifier_expr),
        // context("path", parse_path_expr),
        context("fn_call", parse_fn_call_expr),
        // context("member", parse_member_expr),
        // context("method_call", parse_method_call_expr),
        context("literal", parse_literal_expr),
        context("var", parse_var_expr),
    ))
    .parse(input)
}

pub(crate) fn parse_parenthesised_expr<'a>(input: Input<'a>) -> PResult<'a, OffsetExpr> {
    delimited(multispace, delimited(tag("("), parse_expression, tag(")")), multispace).parse(input)
}

pub(crate) fn parse_quantifier_expr<'a>(input: Input<'a>) -> PResult<'a, OffsetExpr> {
    let prev_offset = input.len();
    let (input, quantifier_kind) = parse_identifier(input)?;

    let quantifier = match quantifier_kind {
        "forall" => Quantifier::Forall,
        "exists" => Quantifier::Exists,
        _ => return Err(nom::Err::Error(Error { parser_errors: vec![], contexts: vec![] })),
    };

    let (input, _) = multispace(input)?;
    // TODO: better `space` management
    let (input, (name, expr)) = delimited(
        tag("("),
        cut(pair(
            delimited(
                delimited(multispace, tag("|"), multispace),
                parse_identifier,
                delimited(multispace, tag("|"), multispace),
            ),
            delimited(multispace, parse_expression, multispace),
        )),
        tag(")"),
    )
    .parse(input)?;
    let after_offset = input.len();

    Ok((
        input,
        build_expr(
            prev_offset,
            after_offset,
            ExprF::Quantified { quantifier, name: name.to_string(), expr },
        ),
    ))
}

pub(crate) fn parse_fn_call_expr<'a>(input: Input<'a>) -> PResult<'a, OffsetExpr> {
    let prev_offset = input.len();
    let (input, name) = context("fn_call name", parse_identifier).parse(input)?;

    let (input, params) = delimited(
        tag("("),
        cut(separated_list0(pair(tag(","), multispace), parse_expression)),
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

pub(crate) fn parse_literal_expr<'a>(input: Input<'a>) -> PResult<'a, OffsetExpr> {
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

pub(crate) fn parse_bool<'a>(input: Input<'a>) -> PResult<'a, bool> {
    alt((map(tag("true"), |_| true), map(tag("false"), |_| false))).parse(input)
}

pub(crate) fn parse_int<'a>(input: Input<'a>) -> PResult<'a, BigInt> {
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

pub(crate) fn parse_sign<'a>(input: Input<'a>) -> PResult<'a, bool> {
    let (input, opt_sign) = opt(alt((
        //
        value(false, tag(&b"-"[..])),
        value(true, tag(&b"+"[..])),
    )))
    .parse(input)?;
    let sign = opt_sign.unwrap_or(true);

    Ok((input, sign))
}

pub(crate) fn parse_var_expr<'a>(input: Input<'a>) -> PResult<'a, OffsetExpr> {
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
        Err::Error(Error { parser_errors: vec![ParserError::Oops], contexts: vec![] })
    })?;
    let after_offset = input.len();

    Ok((input, build_expr(prev_offset, after_offset, ExprF::Variable { name: ident.to_string() })))
}

// TODO: parse module references `fv_std::SOMETHING`
pub(crate) fn parse_identifier<'a>(input: Input<'a>) -> PResult<'a, &'a str> {
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

#[cfg(test)]
pub mod tests {
    use noirc_frontend::{
        monomorphization::ast::{
            Expression, FuncId, Function, InlineType, LocalId, Type as NoirType,
        },
        shared::Visibility,
    };

    use crate::{
        Attribute, State,
        ast::{AnnExpr, Literal, cata},
        parse_attribute,
    };

    use super::*;

    pub fn empty_state() -> State<'static> {
        State {
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
                    (
                        LocalId(3),
                        false,
                        "xs".to_string(),
                        NoirType::Array(3, Box::new(NoirType::Field)),
                        Visibility::Public,
                    ),
                    (
                        LocalId(3),
                        false,
                        "user".to_string(),
                        NoirType::Tuple(vec![NoirType::Bool, NoirType::Unit]),
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

    pub fn parse<'a>(input: &'a str) -> PResult<'a, OffsetExpr> {
        parse_expression(input)
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
        dbg!(&expr);
        assert_eq!(expr.0, "");
    }

    #[test]
    fn test_function_call() {
        let expr = parse("banica(1, banica(a, kek))").unwrap();
        dbg!(&expr);
        assert_eq!(expr.0, "");
    }

    #[test]
    fn test_equality_precedence() {
        let expr = parse("a == b | c == d").unwrap();
        let expr_expected = parse("((a == (b | c)) == d)").unwrap();
        dbg!(&expr);
        assert_eq!(expr.0, "");
        assert_eq!(expr_expected.0, "");

        fn strip_ann<T>(expr: AnnExpr<T>) -> AnnExpr<()> {
            cata(expr, &|_, expr| AnnExpr { ann: (), expr: Box::new(expr) })
        }

        let expr_expected_flat: OffsetExpr = cata(expr_expected.1, &|ann, expr| match expr {
            ExprF::Parenthesised { expr } => expr,
            _ => OffsetExpr { ann, expr: Box::new(expr) },
        });

        assert_eq!(strip_ann(expr.1), strip_ann(expr_expected_flat));
    }

    #[test]
    fn test_sum() {
        let identche = "1 + 2 * 3";
        let expr = parse(identche).unwrap();
        dbg!(&expr);
        assert_eq!(expr.0, "");
    }

    #[test]
    fn test_quantifier() {
        let expr = parse("exists(|x| 1 + x)").unwrap();
        dbg!(&expr);
        assert_eq!(expr.0, "");
    }

    #[test]
    fn test_index() {
        let expr = parse("(tutmanik + 3)[12 < 3]").unwrap();
        dbg!(&expr);
        assert_eq!(expr.0, "");
    }

    #[test]
    fn test_member() {
        let expr = parse("ala.0.17").unwrap();
        dbg!(&expr);
        assert_eq!(expr.0, "");
    }

    #[test]
    fn test_postfix() {
        let expr = parse("5 + ala.126[nica].012[1[3]][2].15[da] / 5").unwrap();
        dbg!(&expr);
        assert_eq!(expr.0, "");
    }

    #[test]
    fn test_ghost() {
        let annotation = "ghost";
        let state = empty_state();
        let attribute = parse_attribute(
            annotation,
            Location {
                span: Span::inclusive(0, annotation.len() as u32),
                file: Default::default(),
            },
            state.function,
            state.global_constants,
            state.functions,
        )
        .unwrap();

        assert!(matches!(attribute, Attribute::Ghost));
    }
}
