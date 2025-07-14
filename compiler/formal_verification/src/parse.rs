use noirc_errors::{Location, Span};
use noirc_frontend::monomorphization::ast as mast;
use nom::{
    Err as NomErr, IResult, Parser,
    branch::alt,
    bytes::complete::{tag, take_while, take_while1},
    character::complete::{digit1 as digit, multispace0 as multispace},
    combinator::{cut, map, opt, recognize, value},
    error::context,
    multi::{many0, separated_list0, separated_list1},
    sequence::{delimited, pair, preceded, terminated},
};
use num_bigint::{BigInt, BigUint, Sign};
use std::{collections::BTreeMap, fmt::Debug};

pub mod errors;
use errors::{Error, ParserErrorKind, build_error, expect, map_nom_err};

use crate::{
    Attribute,
    ast::{BinaryOp, ExprF, Literal, OffsetExpr, Quantifier, UnaryOp, Variable},
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
) -> Result<Attribute, Error> {
    // NOTE: #['...]
    //       ^^^^^^^ - received `Location`
    //          ^^^  - relevant stuff
    // TODO: don't do this here
    location = Location {
        span: Span::inclusive(location.span.start() + 3, location.span.end() - 1),
        ..location
    };

    let (i, ident) = match expect("attribute name", parse_identifier)(annotation) {
        Ok(result) => result,
        Err(nom_err) => {
            return match nom_err {
                NomErr::Error(e) | NomErr::Failure(e) => Err(e),
                NomErr::Incomplete(_) => Err(build_error(
                    annotation,
                    ParserErrorKind::Message("Incomplete input".to_string()),
                )),
            };
        }
    };

    match ident {
        "ghost" => {
            if !i.is_empty() {
                return Err(build_error(
                    i,
                    ParserErrorKind::Message(format!(
                        "Unexpected input after 'ghost' attribute: '{}'",
                        i
                    )),
                ));
            }
            Ok(Attribute::Ghost)
        }
        "ensures" | "requires" => {
            let mut expr_parser = delimited(
                preceded(multispace, expect("'('", tag("("))),
                delimited(multispace, parse_expression, multispace),
                cut(expect("')'", tag(")"))),
            );

            match expr_parser.parse(i) {
                Ok((rest, expr)) => {
                    if !rest.is_empty() {
                        return Err(build_error(
                            rest,
                            ParserErrorKind::Message(format!(
                                "Unexpected trailing input: '{}'",
                                rest
                            )),
                        ));
                    }
                    let spanned_expr = span_expr(location, annotation.len() as u32, expr);
                    if ident == "ensures" {
                        Ok(Attribute::Ensures(spanned_expr))
                    } else {
                        Ok(Attribute::Requires(spanned_expr))
                    }
                }
                Err(nom_err) => match nom_err {
                    NomErr::Error(e) | NomErr::Failure(e) => Err(e),
                    NomErr::Incomplete(_) => Err(build_error(
                        i,
                        ParserErrorKind::Message("Incomplete input".to_string()),
                    )),
                },
            }
        }
        unknown => {
            let err_kind = ParserErrorKind::UnknownAttribute(unknown.to_string());
            Err(build_error(annotation, err_kind))
        }
    }
}

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
            many0(pair(
                delimited(multispace, expect("'==>'", tag("==>")), multispace),
                parse_equality_expr,
            )),
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
                delimited(
                    multispace,
                    expect("'==' or '!='".to_string(), alt((tag("=="), tag("!=")))),
                    multispace,
                ),
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
        context(
            "or",
            many0(pair(
                delimited(multispace, expect("'|'".to_string(), tag("|")), multispace),
                parse_and_expr,
            )),
        ),
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
        context(
            "and",
            many0(pair(
                delimited(multispace, expect("'&'".to_string(), tag("&")), multispace),
                parse_xor_expr,
            )),
        ),
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
            many0(pair(
                delimited(multispace, expect("'^'".to_string(), tag("^")), multispace),
                parse_comparison_expr,
            )),
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
    let (i, mut expr_left) = parse_shift_expr(input)?;

    // Comparison operators are not associative (e.g., `a < b < c` is invalid),
    // so we just look for one optional occurrence.
    if let (i, Some((op, expr_right))) = opt(pair(
        delimited(
            multispace,
            expect(
                "'<=', '>=', '<' or '>'".to_string(),
                alt((tag("<="), tag(">="), tag("<"), tag(">"))),
            ),
            multispace,
        ),
        // NOTE: If an operator was found, the RHS is now MANDATORY. `cut` enforces this.
        cut(parse_shift_expr),
    ))
    .parse(i)?
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
        return Ok((i, expr_left));
    }

    Ok((i, expr_left))
}

pub(crate) fn parse_shift_expr<'a>(input: Input<'a>) -> PResult<'a, OffsetExpr> {
    let (i, mut expr_left) = parse_additive_expr(input)?;

    if let (i, Some((op, expr_right))) = opt(pair(
        delimited(multispace, expect("'<<' or '>>'", alt((tag("<<"), tag(">>")))), multispace),
        cut(parse_additive_expr),
    ))
    .parse(i)?
    {
        let op_kind = match op {
            "<<" => BinaryOp::ShiftLeft,
            ">>" => BinaryOp::ShiftRight,
            _ => unreachable!(),
        };
        expr_left = OffsetExpr {
            ann: build_offset_from_exprs(&expr_left, &expr_right),
            expr: Box::new(ExprF::BinaryOp { op: op_kind, expr_left, expr_right }),
        };
        return Ok((i, expr_left));
    }

    Ok((i, expr_left))
}

pub(crate) fn parse_additive_expr<'a>(input: Input<'a>) -> PResult<'a, OffsetExpr> {
    let (mut i, mut expr_left) = parse_multiplicative_expr(input)?;

    loop {
        let (next_i, remainder) = opt(pair(
            delimited(multispace, expect("'+' or '-'", alt((tag("+"), tag("-")))), multispace),
            cut(parse_multiplicative_expr),
        ))
        .parse(i)?;

        if let Some((op, expr_right)) = remainder {
            let op_kind = match op {
                "+" => BinaryOp::Add,
                "-" => BinaryOp::Sub,
                _ => unreachable!(),
            };
            expr_left = OffsetExpr {
                ann: build_offset_from_exprs(&expr_left, &expr_right),
                expr: Box::new(ExprF::BinaryOp { op: op_kind, expr_left, expr_right }),
            };
            i = next_i;
        } else {
            return Ok((i, expr_left));
        }
    }
}

pub(crate) fn parse_multiplicative_expr<'a>(input: Input<'a>) -> PResult<'a, OffsetExpr> {
    let (input, (first, remainder)) = pair(
        parse_prefix_expr,
        context(
            "multiplicative",
            many0(pair(
                delimited(
                    multispace,
                    expect("'*', '/' or '%'".to_string(), alt((tag("*"), tag("/"), tag("%")))),
                    multispace,
                ),
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
            map(
                preceded(
                    terminated(expect("'!'".to_string(), tag("!")), multispace),
                    parse_prefix_expr,
                ),
                |expr| OffsetExpr {
                    ann: expr.ann,
                    expr: Box::new(ExprF::UnaryOp { op: UnaryOp::Not, expr }),
                },
            ),
        ),
        parse_postfix_expr,
    ))
    .parse(input)
}

pub(crate) enum Postfix {
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
                        NomErr::Error(Error { parser_errors: vec![], contexts: vec![] })
                    })?,
                }),
            },
        })
    })?;

    Ok((input, final_expr))
}

pub(crate) fn parse_any_suffix<'a>(input: Input<'a>) -> PResult<'a, Postfix> {
    alt((
        map(parse_index_suffix, Postfix::ArrayIndex),
        map(parse_member_suffix, Postfix::TupleMember),
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

fn trace<'a, P, O: Debug>(
    name: &'static str,
    mut parser: P,
) -> impl FnMut(Input<'a>) -> IResult<Input<'a>, O, Error>
where
    P: Parser<Input<'a>, Output = O, Error = Error>,
{
    move |input: Input<'a>| {
        println!("[trace] Enter '{}' on input: {:?}", name, input);
        let result = parser.parse(input);
        match &result {
            Ok((remaining, output)) => {
                println!(
                    "[trace] Success on '{}'! Output: {:?}, Remaining: {:?}",
                    name, output, remaining
                );
            }
            Err(e) => {
                println!("[trace] Fail on '{}'! Error: {:?}", name, e);
            }
        }
        result
    }
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
        multispace,
        delimited(
            expect("'('".to_string(), tag("(")),
            parse_expression,
            expect("')'".to_string(), tag(")")),
        ),
        multispace,
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
                expect("'|'".to_string(), tag("|")),
                // NOTE: unlike functions, quantifiers need to have at least one bound variable
                cut(separated_list1(
                    delimited(multispace, expect("','".to_string(), tag(",")), multispace),
                    parse_identifier,
                )),
                expect("'|'".to_string(), tag("|")),
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
    let (i, ident) = parse_identifier(input)?;
    match ident {
        "forall" => Ok((i, Quantifier::Forall)),
        "exists" => Ok((i, Quantifier::Exists)),
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
        let ExprF::Variable(Variable { name: i, .. }) = *expr.expr else { panic!() };
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

        let first_error = err.parser_errors.get(0).expect("Should have one parser error");
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

        let first_error = err.parser_errors.get(0).expect("Should have one parser error");
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
