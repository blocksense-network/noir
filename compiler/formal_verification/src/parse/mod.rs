use noirc_errors::Location;
use noirc_frontend::monomorphization::ast::{Expression, Function, GlobalId, Type};
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
use vir::{
    ast::{
        AutospecUsage, CallTarget, CallTargetKind, Constant, Expr, ExprX, FunX, Ident, IntRange,
        Path, PathX, SpannedTyped, Typ, TypX, VarIdent,
    },
    messages::{RawSpan, Span},
};

const SPAN: LazyCell<Span> = LazyCell::new(|| Span {
    raw_span: Arc::new(()),
    id: 0,
    data: vec![],
    as_string: "".to_string(),
});

pub enum Attribute {
    Ghost,
    Ensures(Expr),
    Requires(Expr),
}

// TODO: variable/function rustc ids
// TODO: function signature

pub fn parse_attribute<'a>(
    annotation: &'a str,
    location: Location,
    function: &Function,
    gamma: &BTreeMap<GlobalId, (String, Type, Expression)>,
) -> Result<Attribute, ()> {
    todo!()
}

fn parse_bool(input: &str) -> IResult<&str, bool> {
    alt((map(tag("true"), |_| true), map(tag("false"), |_| false))).parse(input)
}

fn sign(input: &str) -> IResult<&str, bool> {
    let (i, opt_sign) = opt(alt((
        //
        value(false, tag(&b"-"[..])),
        value(true, tag(&b"+"[..])),
    )))
    .parse(input)?;
    let sign = opt_sign.unwrap_or(true);

    Ok((i, sign))
}

fn parse_int(input: &str) -> IResult<&str, BigInt> {
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

fn parse_constant(input: &str) -> IResult<&str, Expr> {
    alt((
        map(parse_bool, |b| {
            SpannedTyped::new(&SPAN, &Arc::new(TypX::Bool), ExprX::Const(Constant::Bool(b)))
        }),
        map(parse_int, |bi| {
            // TODO: Better type than `TypX::Int`
            SpannedTyped::new(
                &SPAN,
                &Arc::new(TypX::Int(IntRange::Int)),
                ExprX::Const(Constant::Int(bi)),
            )
        }),
    ))
    .parse(input)
}

fn parse_identifier(input: &str) -> IResult<&str, &str> {
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

fn parse_var(input: &str) -> IResult<&str, Expr> {
    let (input, ident) = parse_identifier(input)?;

    // TODO: Actual types, fetched from signatures
    Ok((
        input,
        SpannedTyped::new(
            &SPAN,
            &Arc::new(TypX::Bool),
            ExprX::Var(VarIdent(
                Arc::new(ident.to_string()),
                vir::ast::VarIdentDisambiguate::RustcId(0),
            )),
        ),
    ))
}

fn parse_fn_call(input: &str) -> IResult<&str, Expr> {
    let (input, name) = parse_identifier(input)?;

    let (input, params) = delimited(
        tag("("),
        separated_list0(pair(tag(","), opt(tag(" "))), parse_expression),
        tag(")"),
    )
    .parse(input)?;

    dbg!(&name, &params);

    // TODO: Actual types, fetched from signatures
    Ok((
        input,
        SpannedTyped::new(
            &SPAN,
            &Arc::new(TypX::Bool),
            ExprX::Call(
                CallTarget::Fun(
                    CallTargetKind::Static,
                    Arc::new(FunX {
                        path: Path::new(PathX { krate: None, segments: Arc::new(vec![]) }),
                    }),
                    Arc::new(vec![]),
                    Arc::new(vec![]),
                    AutospecUsage::Final,
                ),
                Arc::new(vec![]),
            ),
        ),
    ))
}

fn parse_expression(input: &str) -> IResult<&str, Expr> {
    alt([
        //
        parse_fn_call,
        parse_constant,
        parse_var,
    ])
    .parse(input)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bool_true() {
        let expr = parse_expression("true").unwrap();
        assert_eq!(expr.0, "");
        assert!(matches!(*expr.1.typ, TypX::Bool));
        assert!(matches!(expr.1.x, ExprX::Const(Constant::Bool(true))));
    }

    #[test]
    fn test_int() {
        let chislo = "1234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890";
        let expr = parse_expression(chislo).unwrap();
        assert_eq!(expr.0, "");
        assert!(matches!(*expr.1.typ, TypX::Int(IntRange::Int)));
        let ExprX::Const(Constant::Int(ref bi)) = expr.1.x else { panic!() };
        assert_eq!(bi.to_str_radix(10), chislo);
    }

    #[test]
    fn test_ident() {
        let identche = "Banica_123_";
        let expr = parse_expression(identche).unwrap();
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
        let expr = parse_var(identche).unwrap();
        assert_eq!(expr.0, "");
        dbg!(expr);
    }

    #[test]
    fn test_function_call() {
        let identche = "banica(1, f(), g(1, kek))";
        let expr = parse_expression(identche).unwrap();
        assert_eq!(expr.0, "");
        dbg!(expr);
    }
}
