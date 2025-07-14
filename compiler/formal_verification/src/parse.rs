use noirc_errors::{Location, Span};
use noirc_frontend::{
    ast::IntegerBitSize,
    monomorphization::ast::{self as mast, Type as NoirType},
    shared::Signedness,
};
use nom::{
    Err as NomErr, IResult, Parser,
    branch::alt,
    bytes::complete::{tag, take_while, take_while1},
    character::complete::{digit1 as digit, multispace0 as multispace, multispace1},
    combinator::{cut, map, not, opt, recognize, value},
    error::context,
    multi::{many0, separated_list0, separated_list1},
    sequence::{delimited, pair, preceded, terminated},
};
use num_bigint::{BigInt, BigUint, Sign};
use std::collections::BTreeMap;

pub mod errors;
use errors::{
    Error, ParserError, ParserErrorKind, ParserErrorWithLocation, build_error, expect, map_nom_err,
};

use crate::{
    Attribute,
    ast::{BinaryOp, ExprF, Identifier, Literal, OffsetExpr, Quantifier, UnaryOp, Variable},
    span_expr,
};

pub type Input<'a> = &'a str;
pub type PResult<'a, T> = IResult<Input<'a>, T, Error>;

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

pub fn parse_attribute<'a>(
    annotation: &'a str,
    mut location: Location,
    _function: &'a mast::Function,
    _global_constants: &'a BTreeMap<mast::GlobalId, (String, mast::Type, mast::Expression)>,
    _functions: &'a BTreeMap<mast::FuncId, mast::Function>,
) -> Result<Attribute, (Vec<ParserErrorWithLocation>, Vec<String>)> {
    // NOTE: #['...]
    //       ^^^^^^^ - received `Location`
    //          ^^^  - relevant stuff
    // TODO: don't do this here
    location = Location {
        span: Span::inclusive(location.span.start() + 3, location.span.end() - 1),
        ..location
    };

    let locate_parser_error = |parser_error: ParserError| ParserErrorWithLocation {
        location: build_location(
            location,
            annotation.len() as u32,
            parser_error.offset,
            parser_error.offset + 1,
        ),
        kind: parser_error.kind.clone(),
    };

    let convert_nom_error = |nom_err: Error| -> (Vec<ParserErrorWithLocation>, Vec<String>) {
        (nom_err.parser_errors.into_iter().map(locate_parser_error).collect(), nom_err.contexts)
    };

    let (input, ident) = match expect("attribute name", parse_identifier)(annotation) {
        Ok(result) => result,
        Err(nom_err) => {
            return Err(convert_nom_error(match nom_err {
                NomErr::Error(e) | NomErr::Failure(e) => e,
                NomErr::Incomplete(_) => build_error(
                    annotation,
                    ParserErrorKind::Message("Incomplete input".to_string()),
                ),
            }));
        }
    };

    match ident {
        "ghost" => {
            if !input.is_empty() {
                return Err(convert_nom_error(build_error(
                    input,
                    ParserErrorKind::Message(format!(
                        "Unexpected input after 'ghost' attribute: '{}'",
                        input
                    )),
                )));
            }
            Ok(Attribute::Ghost)
        }
        "ensures" | "requires" => {
            let mut expr_parser = delimited(
                preceded(multispace, expect("'('", tag("("))),
                delimited(multispace, parse_expression, multispace),
                cut(expect("')'", tag(")"))),
            );

            match expr_parser.parse(input) {
                Ok((rest, expr)) => {
                    if !rest.is_empty() {
                        return Err(convert_nom_error(build_error(
                            rest,
                            ParserErrorKind::Message(format!(
                                "Unexpected trailing input: '{}'",
                                rest
                            )),
                        )));
                    }
                    let spanned_expr = span_expr(location, annotation.len() as u32, expr);
                    if ident == "ensures" {
                        Ok(Attribute::Ensures(spanned_expr))
                    } else {
                        Ok(Attribute::Requires(spanned_expr))
                    }
                }
                Err(nom_err) => match nom_err {
                    NomErr::Error(e) | NomErr::Failure(e) => Err(convert_nom_error(e)),
                    NomErr::Incomplete(_) => Err(convert_nom_error(build_error(
                        input,
                        ParserErrorKind::Message("Incomplete input".to_string()),
                    ))),
                },
            }
        }
        unknown => {
            let err_kind = ParserErrorKind::UnknownAttribute(unknown.to_string());
            Err(convert_nom_error(build_error(annotation, err_kind)))
        }
    }
}

pub(crate) fn parse_expression<'a>(input: Input<'a>) -> PResult<'a, OffsetExpr> {
    alt((
        //
        parse_implication_expr,
    ))
    .parse(input)
}

pub(crate) fn parse_implication_expr<'a>(input: Input<'a>) -> PResult<'a, OffsetExpr> {
    let (mut input, mut expr_left) = parse_equality_expr(input)?;
    loop {
        let (next_input, remainder) = opt(context(
            "implication",
            pair(
                delimited(
                    multispace,
                    expect("'==>'", tag("==>").map(|_| BinaryOp::Implies)),
                    multispace,
                ),
                cut(parse_equality_expr),
            ),
        ))
        .parse(input)?;

        if let Some((op, expr_right)) = remainder {
            expr_left = OffsetExpr {
                ann: build_offset_from_exprs(&expr_left, &expr_right),
                expr: Box::new(ExprF::BinaryOp { op, expr_left, expr_right }),
            };
            input = next_input;
        } else {
            return Ok((input, expr_left));
        }
    }
}

pub(crate) fn parse_equality_expr<'a>(input: Input<'a>) -> PResult<'a, OffsetExpr> {
    let (mut input, mut expr_left) = parse_or_expr(input)?;

    loop {
        let (next_input, remainder) = opt(
            // Re-introduce `context` here to wrap the operator and RHS parser
            context(
                "equality expression", // The context name for error messages
                pair(
                    delimited(
                        multispace,
                        expect(
                            "'==' or '!='",
                            alt((
                                // NOTE: We need to check that the immediate next character after
                                //       the `==` is not `>`, otherwise, we'll `cut` into trying to
                                //       parse it a normal expression while it should actually be
                                //       parsed as implication (`==>`)
                                terminated(tag("=="), not(tag(">"))).map(|_| BinaryOp::Eq),
                                tag("!=").map(|_| BinaryOp::Neq),
                            )),
                        ),
                        multispace,
                    ),
                    cut(parse_or_expr),
                ),
            ),
        )
        .parse(input)?;

        if let Some((op, expr_right)) = remainder {
            expr_left = OffsetExpr {
                ann: build_offset_from_exprs(&expr_left, &expr_right),
                expr: Box::new(ExprF::BinaryOp { op, expr_left, expr_right }),
            };
            input = next_input;
        } else {
            return Ok((input, expr_left));
        }
    }
}

pub(crate) fn parse_or_expr<'a>(input: Input<'a>) -> PResult<'a, OffsetExpr> {
    let (mut input, mut expr_left) = parse_and_expr(input)?;
    loop {
        let (next_input, remainder) = opt(context(
            "or",
            pair(
                delimited(multispace, expect("'|'", tag("|").map(|_| BinaryOp::Or)), multispace),
                cut(parse_and_expr),
            ),
        ))
        .parse(input)?;

        if let Some((op, expr_right)) = remainder {
            expr_left = OffsetExpr {
                ann: build_offset_from_exprs(&expr_left, &expr_right),
                expr: Box::new(ExprF::BinaryOp { op, expr_left, expr_right }),
            };
            input = next_input;
        } else {
            return Ok((input, expr_left));
        }
    }
}

pub(crate) fn parse_and_expr<'a>(input: Input<'a>) -> PResult<'a, OffsetExpr> {
    let (mut input, mut expr_left) = parse_xor_expr(input)?;
    loop {
        let (next_input, remainder) = opt(context(
            "and",
            pair(
                delimited(multispace, expect("'&'", tag("&").map(|_| BinaryOp::And)), multispace),
                cut(parse_xor_expr),
            ),
        ))
        .parse(input)?;

        if let Some((op, expr_right)) = remainder {
            expr_left = OffsetExpr {
                ann: build_offset_from_exprs(&expr_left, &expr_right),
                expr: Box::new(ExprF::BinaryOp { op, expr_left, expr_right }),
            };
            input = next_input;
        } else {
            return Ok((input, expr_left));
        }
    }
}

pub(crate) fn parse_xor_expr<'a>(input: Input<'a>) -> PResult<'a, OffsetExpr> {
    let (mut input, mut expr_left) = parse_comparison_expr(input)?;
    loop {
        let (next_input, remainder) = opt(context(
            "xor",
            pair(
                delimited(multispace, expect("'^'", tag("^").map(|_| BinaryOp::Xor)), multispace),
                cut(parse_comparison_expr),
            ),
        ))
        .parse(input)?;

        if let Some((op, expr_right)) = remainder {
            expr_left = OffsetExpr {
                ann: build_offset_from_exprs(&expr_left, &expr_right),
                expr: Box::new(ExprF::BinaryOp { op, expr_left, expr_right }),
            };
            input = next_input;
        } else {
            return Ok((input, expr_left));
        }
    }
}
pub(crate) fn parse_comparison_expr<'a>(input: Input<'a>) -> PResult<'a, OffsetExpr> {
    let (input, mut expr_left) = parse_shift_expr(input)?;

    // Comparison operators are not associative (e.g., `a < b < c` is invalid),
    // so we just look for one optional occurrence.
    if let (input, Some((op, expr_right))) = opt(pair(
        delimited(
            multispace,
            expect(
                "'<=', '>=', '<' or '>'".to_string(),
                alt((
                    tag("<=").map(|_| BinaryOp::Le),
                    tag(">=").map(|_| BinaryOp::Ge),
                    tag("<").map(|_| BinaryOp::Lt),
                    tag(">").map(|_| BinaryOp::Gt),
                )),
            ),
            multispace,
        ),
        // NOTE: If an operator was found, the RHS is now MANDATORY. `cut` enforces this.
        cut(parse_shift_expr),
    ))
    .parse(input)?
    {
        expr_left = OffsetExpr {
            ann: build_offset_from_exprs(&expr_left, &expr_right),
            expr: Box::new(ExprF::BinaryOp { op, expr_left, expr_right }),
        };
        return Ok((input, expr_left));
    }

    Ok((input, expr_left))
}

pub(crate) fn parse_shift_expr<'a>(input: Input<'a>) -> PResult<'a, OffsetExpr> {
    let (input, mut expr_left) = parse_additive_expr(input)?;

    if let (input, Some((op, expr_right))) = opt(pair(
        delimited(
            multispace,
            expect(
                "'<<' or '>>'",
                alt((
                    tag("<<").map(|_| BinaryOp::ShiftLeft),
                    tag(">>").map(|_| BinaryOp::ShiftRight),
                )),
            ),
            multispace,
        ),
        cut(parse_additive_expr),
    ))
    .parse(input)?
    {
        expr_left = OffsetExpr {
            ann: build_offset_from_exprs(&expr_left, &expr_right),
            expr: Box::new(ExprF::BinaryOp { op, expr_left, expr_right }),
        };
        return Ok((input, expr_left));
    }

    Ok((input, expr_left))
}

pub(crate) fn parse_additive_expr<'a>(input: Input<'a>) -> PResult<'a, OffsetExpr> {
    let (mut input, mut expr_left) = parse_multiplicative_expr(input)?;

    loop {
        let (next_input, remainder) = opt(pair(
            delimited(
                multispace,
                expect(
                    "'+' or '-'",
                    alt((tag("+").map(|_| BinaryOp::Add), tag("-").map(|_| BinaryOp::Sub))),
                ),
                multispace,
            ),
            cut(parse_multiplicative_expr),
        ))
        .parse(input)?;

        if let Some((op, expr_right)) = remainder {
            expr_left = OffsetExpr {
                ann: build_offset_from_exprs(&expr_left, &expr_right),
                expr: Box::new(ExprF::BinaryOp { op, expr_left, expr_right }),
            };
            input = next_input;
        } else {
            return Ok((input, expr_left));
        }
    }
}

pub(crate) fn parse_multiplicative_expr<'a>(input: Input<'a>) -> PResult<'a, OffsetExpr> {
    let (mut input, mut expr_left) = parse_prefix_expr(input)?;

    loop {
        let (next_input, remainder) = opt(pair(
            delimited(
                multispace,
                expect(
                    "'*', '/', or '%'",
                    alt((
                        tag("*").map(|_| BinaryOp::Mul),
                        tag("/").map(|_| BinaryOp::Div),
                        tag("%").map(|_| BinaryOp::Mod),
                    )),
                ),
                multispace,
            ),
            cut(parse_prefix_expr),
        ))
        .parse(input)?;

        if let Some((op, expr_right)) = remainder {
            expr_left = OffsetExpr {
                ann: build_offset_from_exprs(&expr_left, &expr_right),
                expr: Box::new(ExprF::BinaryOp { op, expr_left, expr_right }),
            };
            input = next_input;
        } else {
            return Ok((input, expr_left));
        }
    }
}

pub(crate) enum Prefix {
    Not,
    // Dereference,
}

pub(crate) fn parse_prefix_expr<'a>(input: Input<'a>) -> PResult<'a, OffsetExpr> {
    let (input, prefixes) = context("prefix", many0(parse_any_prefix)).parse(input)?;

    let (input, base_expr) = parse_postfix_expr(input)?;

    let final_expr = prefixes.into_iter().rev().fold(base_expr, |inner_expr, prefix| {
        // TODO: real span
        let ann = inner_expr.ann;
        let expr_f = match prefix {
            Prefix::Not => ExprF::UnaryOp { op: UnaryOp::Not, expr: inner_expr },
        };
        OffsetExpr { ann, expr: Box::new(expr_f) }
    });

    Ok((input, final_expr))
}

pub(crate) fn parse_any_prefix<'a>(input: Input<'a>) -> PResult<'a, Prefix> {
    alt((
        //
        context("not", terminated(expect("'!'", tag("!")), multispace).map(|_| Prefix::Not)),
    ))
    .parse(input)
}

pub(crate) enum Postfix {
    ArrayIndex(OffsetExpr),
    TupleMember(BigInt),
    Cast(CastTargetType),
}

pub(crate) enum CastTargetType {
    Field,
    Integer(Signedness, IntegerBitSize),
    Bool,
}

pub(crate) fn parse_postfix_expr<'a>(input: Input<'a>) -> PResult<'a, OffsetExpr> {
    let (mut input, mut expr_base) = parse_atom_expr(input)?;

    loop {
        // `cut` ensures that if we see a `.` or `[` but the rest is invalid, we get a hard error.
        let (next_input, suffix) = cut(opt(parse_any_suffix)).parse(input)?;

        if let Some(s) = suffix {
            expr_base = match s {
                Postfix::ArrayIndex(index_expr) => {
                    let ann = build_offset_from_exprs(&expr_base, &index_expr);
                    OffsetExpr {
                        ann,
                        expr: Box::new(ExprF::Index { expr: expr_base, index: index_expr }),
                    }
                }
                Postfix::TupleMember(index) => {
                    let index_u32 = index.try_into().map_err(|_| {
                        NomErr::Error(build_error(input, ParserErrorKind::InvalidTupleIndex))
                    })?;
                    let ann = (expr_base.ann.0, (input.len() - next_input.len()) as u32); // Approximate span
                    OffsetExpr {
                        ann,
                        expr: Box::new(ExprF::TupleAccess { expr: expr_base, index: index_u32 }),
                    }
                }
                Postfix::Cast(target_type) => {
                    let ann = (expr_base.ann.0, (input.len() - next_input.len()) as u32); // Approximate span
                    OffsetExpr {
                        ann,
                        expr: Box::new(ExprF::Cast {
                            expr: expr_base,
                            target: match target_type {
                                CastTargetType::Field => NoirType::Field,
                                CastTargetType::Integer(s, b) => NoirType::Integer(s, b),
                                CastTargetType::Bool => NoirType::Bool,
                            },
                        }),
                    }
                }
            };
            input = next_input;
        } else {
            return Ok((input, expr_base));
        }
    }
}

pub(crate) fn parse_any_suffix<'a>(input: Input<'a>) -> PResult<'a, Postfix> {
    alt((
        context("index", map(parse_index_suffix, Postfix::ArrayIndex)),
        context("member", map(parse_member_suffix, Postfix::TupleMember)),
        context("cast", map(parse_cast_suffix, Postfix::Cast)),
    ))
    .parse(input)
}

pub(crate) fn parse_index_suffix<'a>(input: Input<'a>) -> PResult<'a, OffsetExpr> {
    preceded(
        multispace,
        delimited(
            expect("'[".to_string(), tag("[")),
            cut(delimited(multispace, parse_expression, multispace)),
            expect("']'".to_string(), tag("]")),
        ),
    )
    .parse(input)
}

pub(crate) fn parse_member_suffix<'a>(input: Input<'a>) -> PResult<'a, BigInt> {
    preceded(pair(multispace, expect("'.' for tuple access".to_string(), tag("."))), cut(parse_int))
        .parse(input)
}

pub(crate) fn parse_cast_suffix<'a>(input: Input<'a>) -> PResult<'a, CastTargetType> {
    let (input, type_ident) = preceded(
        expect("'as'", delimited(multispace1, tag("as"), multispace1)),
        cut(parse_identifier),
    )
    .parse(input)?;

    if type_ident == "Field" {
        return Ok((input, CastTargetType::Field));
    }

    if type_ident == "bool" {
        return Ok((input, CastTargetType::Bool));
    }

    if let Some((signedness @ ("i" | "u"), size)) = type_ident.split_at_checked(1) {
        return Ok((
            input,
            CastTargetType::Integer(
                match signedness {
                    "i" => Signedness::Signed,
                    "u" => Signedness::Unsigned,
                    _ => unreachable!(),
                },
                match size {
                    "1" => IntegerBitSize::One,
                    "8" => IntegerBitSize::Eight,
                    "16" => IntegerBitSize::Sixteen,
                    "32" => IntegerBitSize::ThirtyTwo,
                    "64" => IntegerBitSize::SixtyFour,
                    "128" => IntegerBitSize::HundredTwentyEight,
                    _ => {
                        return Err(NomErr::Error(build_error(
                            size,
                            ParserErrorKind::InvalidIntegerLiteral,
                        )));
                    }
                },
            ),
        ));
    }

    Err(NomErr::Error(build_error(type_ident, ParserErrorKind::InvalidIntegerLiteral)))
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
    delimited(
        expect("'('".to_string(), tag("(")),
        delimited(multispace, parse_expression, multispace),
        expect("')'".to_string(), tag(")")),
    )
    .parse(input)
}

pub(crate) fn parse_quantifier_expr<'a>(input: Input<'a>) -> PResult<'a, OffsetExpr> {
    let prev_offset = input.len();
    let (input, quantifier) = context("quantifier keyword", parse_quantifier_kind).parse(input)?;

    let (input, _) = multispace(input)?;
    let (input, (variables, expr)) = delimited(
        expect("'('".to_string(), tag("(")),
        cut(pair(
            delimited(
                expect("'|'".to_string(), pair(tag("|"), multispace)),
                // NOTE: unlike functions, quantifiers need to have at least one bound variable
                cut(separated_list1(
                    delimited(multispace, expect("','".to_string(), tag(",")), multispace),
                    parse_identifier,
                )),
                expect("'|'".to_string(), pair(multispace, tag("|"))),
            ),
            delimited(multispace, parse_expression, multispace),
        )),
        expect("')'".to_string(), tag(")")),
    )
    .parse(input)?;
    let after_offset = input.len();

    Ok((
        input,
        build_expr(
            prev_offset,
            after_offset,
            ExprF::Quantified {
                quantifier,
                variables: variables
                    .into_iter()
                    .map(|name| Variable { name: name.to_string(), id: None })
                    .collect(),
                expr,
            },
        ),
    ))
}

fn parse_quantifier_kind<'a>(input: Input<'a>) -> PResult<'a, Quantifier> {
    let (input, ident) = parse_identifier(input)?;
    match ident {
        "forall" => Ok((input, Quantifier::Forall)),
        "exists" => Ok((input, Quantifier::Exists)),
        // NOTE: If the identifier is not a valid quantifier, fail with a specific error
        _ => Err(NomErr::Error(build_error(
            input,
            ParserErrorKind::InvalidQuantifier(ident.to_string()),
        ))),
    }
}

pub(crate) fn parse_fn_call_expr<'a>(input: Input<'a>) -> PResult<'a, OffsetExpr> {
    let prev_offset = input.len();
    let (input, name) = context("fn_call name", parse_identifier).parse(input)?;

    let (input, params) = delimited(
        expect("'('".to_string(), tag("(")),
        cut(separated_list0(
            delimited(multispace, expect("','".to_string(), tag(",")), multispace),
            parse_expression,
        )),
        expect("')'", tag(")")),
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
    expect("boolean 'true' or 'false'", alt((value(true, tag("true")), value(false, tag("false")))))
        .parse(input)
}

pub(crate) fn parse_int<'a>(input: Input<'a>) -> PResult<'a, BigInt> {
    let (input, sign) = parse_sign(input)?;

    let (input, digits) = expect("an integer".to_string(), digit)(input)?;

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
    let (input, opt_sign) = opt(map_nom_err(
        alt((value(false, tag("-")), value(true, tag("+")))),
        |_| ParserErrorKind::Message("Should not fail".to_string()), //  NOTE: `opt` makes this effectively infallible
    ))
    .parse(input)?;
    let sign = opt_sign.unwrap_or(true);
    Ok((input, sign))
}

const FORBIDDEN_IDENTIFIERS: &[&str] = &["forall", "exists"];

// TODO: parse module references `fv_std::SOMETHING`
pub(crate) fn parse_var_expr<'a>(input: Input<'a>) -> PResult<'a, OffsetExpr> {
    let prev_offset = input.len();

    let (input, ident) = parse_identifier(input)?;

    let after_offset = input.len();

    if FORBIDDEN_IDENTIFIERS.contains(&ident) {
        return Err(NomErr::Error(build_error(
            input,
            ParserErrorKind::InvalidIdentifier(ident.to_string()),
        )));
    }

    Ok((
        input,
        build_expr(
            prev_offset,
            after_offset,
            ExprF::Variable(Variable { name: ident.to_string(), id: None }),
        ),
    ))
}

pub(crate) fn parse_identifier<'a>(input: Input<'a>) -> PResult<'a, &'a str> {
    fn is_valid_start(c: char) -> bool {
        c.is_ascii_alphabetic() || c == '_'
    }

    fn is_valid_char(c: char) -> bool {
        c.is_ascii_alphanumeric() || c == '_'
    }

    let identifier_parser = recognize(pair(take_while1(is_valid_start), take_while(is_valid_char)));

    expect("an identifier".to_string(), identifier_parser)(input)
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
        ast::{Literal, cata, strip_ann},
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
                        LocalId(4),
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
            min_local_id: Box::leak(Box::new(5)),
        }
    }

    pub fn parse<'a>(input: &'a str) -> PResult<'a, OffsetExpr> {
        parse_expression(input)
    }

    pub fn test_precedence_equality(raw: &str, parenthesised: &str) {
        let expr = parse(raw).unwrap();
        let expr_expected = parse(parenthesised).unwrap();
        dbg!(&strip_ann(expr.1.clone()));
        assert_eq!(expr.0, "");
        assert_eq!(expr_expected.0, "");

        let expr_expected_flat: OffsetExpr = cata(expr_expected.1, &|ann, expr| match expr {
            ExprF::Parenthesised { expr } => expr,
            _ => OffsetExpr { ann, expr: Box::new(expr) },
        });

        assert_eq!(strip_ann(expr.1), strip_ann(expr_expected_flat));
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
        let ExprF::Variable(Variable { name: input, .. }) = *expr.expr else { panic!() };
        assert_eq!(&input, identche);
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
        test_precedence_equality("a == b | c == d", "((a == (b | c)) == d)");
    }

    #[test]
    fn test_and_precedence() {
        test_precedence_equality(
            "exists(|i| (0 <= i) & (i < 20) & arr[i] > 100)",
            "exists(|i| (((0 <= i) & (i < 20)) & (arr[i] > 100)))",
            // "exists(|i| ((((0 <= i) & (i < 20)) & arr[i]) > 100))",
        );
    }

    #[test]
    fn test_sum() {
        let identche = "1 + 2 * 3";
        let expr = parse(identche).unwrap();
        dbg!(&expr);
        assert_eq!(expr.0, "");
    }

    #[test]
    fn test_parenthesis() {
        let s = "( (( 1 + 2 ) ) * 3   )";
        let expr = parse(s).unwrap();
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
    fn test_prefix() {
        let expr = parse("(!1 + !2 * !x)").unwrap();
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

    #[test]
    fn test_bitshift() {
        let annotation = "ensures(result == x >> y)";
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

        let Attribute::Ensures(expr) = attribute else { panic!() };
        dbg!(strip_ann(expr));
    }

    #[test]
    fn test_cast() {
        let annotation = "ensures((15 as i16 - 3 > 2) & ((result as Field - 6) as u64 == 1 + a as u64 >> kek as u8))";
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

        let Attribute::Ensures(expr) = attribute else { panic!() };
        dbg!(strip_ann(expr));
    }

    #[test]
    fn test_parse_failure_identifier() {
        let annotation = "ensures(5 > 'invalid)";
        let state = empty_state();
        let result = parse_attribute(
            annotation,
            Location {
                span: Span::inclusive(0, annotation.len() as u32),
                file: Default::default(),
            },
            state.function,
            state.global_constants,
            state.functions,
        );

        let err = result.unwrap_err();

        let first_error = err.0.get(0).expect("Should have one parser error");
        assert!(matches!(&first_error.kind, ParserErrorKind::Expected { expected, found }
            if expected == "an identifier" && found.starts_with("'invalid")
        ));
    }

    #[test]
    fn test_parse_failure_attribute() {
        let annotation = "banica(true)";
        let state = empty_state();
        let result = parse_attribute(
            annotation,
            Location {
                span: Span::inclusive(0, annotation.len() as u32),
                file: Default::default(),
            },
            state.function,
            state.global_constants,
            state.functions,
        );

        let err = result.unwrap_err();

        let first_error = err.0.get(0).expect("Should have one parser error");
        assert!(matches!(&first_error.kind, ParserErrorKind::UnknownAttribute(attr)
            if attr == "banica"
        ));
    }

    #[test]
    fn test_parse_failure_salvo() {
        let annotations = vec![
            // Valid annotation, invalid quantifier
            "requires(exists(x > 5))",
            "requires(forall(|| x > 5))",
            "requires(forall(||))",
            "requires(forall)",
            "requires(forall(|i x > 5))",
            "requires(exists |j| x < 4)",
            "requires(exists())",
            // Invalid annotation
            "ensures(result > 5",
            "ensures result > 5)",
            "ensures result > 5",
            "requires",
            "ensures",
            "ensures()",
            "ensures(result > 4)x",
            "requires x == 4)",
        ];
        for annotation in annotations {
            let state = empty_state();
            let result = parse_attribute(
                annotation,
                Location {
                    span: Span::inclusive(0, annotation.len() as u32),
                    file: Default::default(),
                },
                state.function,
                state.global_constants,
                state.functions,
            );

            let err = result.unwrap_err();
            dbg!(annotation, err);
        }
    }
}
