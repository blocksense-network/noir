use noirc_errors::{Location, Span};
use noirc_evaluator::vir::vir_gen::{
    Attribute, build_span_no_id,
    expr_to_vir::{expr::function_name_to_vir_fun, types::ast_type_to_vir_type},
};
use noirc_frontend::{
    hir::resolution::errors::ResolverError,
    monomorphization::{FUNC_RETURN_VAR_NAME, ast::{Expression, FuncId, Function, GlobalId, Type}},
    parser::{ParserError as NoirParserError, ParserErrorReason},
};
use nom::{
    Err, Finish, IResult, Parser,
    branch::alt,
    bytes::complete::{tag, take_while, take_while1},
    character::complete::{digit1 as digit, multispace0 as multispace, space0 as space},
    combinator::{cut, map, opt, recognize, value},
    error::{ContextError, ErrorKind, ParseError, context},
    multi::{fold_many0, many0, separated_list0, separated_list1},
    sequence::{delimited, pair, preceded, separated_pair, terminated},
};
use num_bigint::{BigInt, BigUint, Sign};
use std::{collections::BTreeMap, fmt::Debug, sync::Arc};
use vir::ast::{
    ArithOp, AutospecUsage, BinaryOp, CallTarget, CallTargetKind, Constant, Expr, ExprX, FunX,
    Ident, IntRange, Mode, Path, PathX, SpannedTyped, Typ, TypX, UnaryOp, VarIdent,
    VarIdentDisambiguate,
};
use vir::messages::Span as VirSpan;

#[derive(Debug)]
struct State<'a> {
    // span: &'a Span,
    full_length: u32,
    location: Location,
    function: &'a Function,
    global_constants: &'a BTreeMap<GlobalId, (String, Type, Expression)>,
    functions: &'a BTreeMap<FuncId, Function>,
}

#[derive(Clone, Debug)]
struct Input<'a, 'b>(&'a str, &'b State<'b>);

impl<'a, 'b> nom::Offset for Input<'a, 'b> {
    fn offset(&self, second: &Self) -> usize {
        <&'a str as nom::Offset>::offset(&self.0, &second.0)
    }
}

impl<'a, 'b> nom::Compare<&'a [u8]> for Input<'a, 'b> {
    fn compare(&self, t: &'a [u8]) -> nom::CompareResult {
        <&'a str as nom::Compare<&'a [u8]>>::compare(&self.0, t)
    }

    fn compare_no_case(&self, t: &'a [u8]) -> nom::CompareResult {
        <&'a str as nom::Compare<&'a [u8]>>::compare_no_case(&self.0, t)
    }
}

impl<'a, 'b> nom::Compare<&'a str> for Input<'a, 'b> {
    fn compare(&self, t: &'a str) -> nom::CompareResult {
        <&'a str as nom::Compare<&'a str>>::compare(&self.0, t)
    }

    fn compare_no_case(&self, t: &'a str) -> nom::CompareResult {
        <&'a str as nom::Compare<&'a str>>::compare_no_case(&self.0, t)
    }
}

impl<'a, 'b> nom::Input for Input<'a, 'b> {
    type Item = <&'a str as nom::Input>::Item;

    type Iter = <&'a str as nom::Input>::Iter;

    type IterIndices = <&'a str as nom::Input>::IterIndices;

    fn input_len(&self) -> usize {
        <&'a str as nom::Input>::input_len(&self.0)
    }

    fn take(&self, index: usize) -> Self {
        let s = <&'a str as nom::Input>::take(&self.0, index);
        Self(s, self.1)
    }

    fn take_from(&self, index: usize) -> Self {
        let s = <&'a str as nom::Input>::take_from(&self.0, index);
        Self(s, self.1)
    }

    fn take_split(&self, index: usize) -> (Self, Self) {
        let (l, r) = <&'a str as nom::Input>::take_split(&self.0, index);
        (Self(l, self.1), Self(r, self.1))
    }

    fn position<P>(&self, predicate: P) -> Option<usize>
    where
        P: Fn(Self::Item) -> bool,
    {
        <&'a str as nom::Input>::position(&self.0, predicate)
    }

    fn iter_elements(&self) -> Self::Iter {
        <&'a str as nom::Input>::iter_elements(&self.0)
    }

    fn iter_indices(&self) -> Self::IterIndices {
        <&'a str as nom::Input>::iter_indices(&self.0)
    }

    fn slice_index(&self, count: usize) -> Result<usize, nom::Needed> {
        <&'a str as nom::Input>::slice_index(&self.0, count)
    }
}

#[derive(Debug)]
struct Error {
    resolver_errors: Vec<ResolverError>,
    location: Location,
}

type PResult<'a, 'b, T> = IResult<Input<'a, 'b>, T, Error>;

impl<'a, 'b> ParseError<Input<'a, 'b>> for Error {
    fn from_error_kind(Input(input, state): Input<'a, 'b>, kind: ErrorKind) -> Self {
        Self {
            resolver_errors: vec![ResolverError::ParserError(Box::new(NoirParserError::empty(
                noirc_frontend::token::Token::Dot,
                // FIXME: whole thing
                state.location,
            )))],
            // NOTE: 1 char
            location: build_location(state, input.len(), input.len()),
            // location: Location {
            //     span: Span::inclusive(
            //         state.location.span.start() + state.full_length - input.len() as u32,
            //         state.location.span.end() + state.full_length - input.len() as u32 + 1,
            //     ),
            //     file: state.location.file,
            // },
        }
    }

    fn append(Input(input, state): Input<'a, 'b>, kind: ErrorKind, mut other: Self) -> Self {
        // TODO: umni neshta
        other.resolver_errors.push(ResolverError::ParserError(Box::new(NoirParserError::empty(
            noirc_frontend::token::Token::Caret,
            build_location(state, input.len(), input.len()),
        ))));
        other
    }
}

// https://github.com/rust-bakery/nom/blob/main/doc/error_management.md

fn build_location(state: &State, prev_offset: usize, after_offset: usize) -> Location {
    Location {
        span: Span::inclusive(
            state.location.span.start() + state.full_length - prev_offset as u32,
            state.location.span.start() + state.full_length - after_offset as u32,
        ),
        file: state.location.file,
    }
}

fn build_span(state: &State, prev_offset: usize, after_offset: usize) -> VirSpan {
    build_span_no_id(
        // TODO: smart debug info from `nom`?
        "".to_string(),
        Some(build_location(state, prev_offset, after_offset)),
    )
}

fn build_expr(
    state: &State,
    prev_offset: usize,
    after_offset: usize,
    atypx: Arc<TypX>,
    exprx: ExprX,
) -> Expr {
    let span = build_span(state, prev_offset, after_offset);

    SpannedTyped::new(&span, &atypx, exprx)
}

fn build_span_from_exprs(state: &State, left: &Expr, right: &Expr) -> VirSpan {
    fn convert_span(input: &str) -> Option<(u32, u32, usize)> {
        if input.is_empty() {
            // Input is empty, cannot decode a span
            return None;
        }

        let trimmed = input.trim_matches(|c| c == '(' || c == ')');
        let parts: Vec<&str> = trimmed.split(',').map(str::trim).collect();

        if parts.len() != 3 {
            // Span must have exactly three components: start, end, file_id
            return None;
        }

        let start_byte = parts[0].parse::<u32>().ok()?;
        let final_byte = parts[1].parse::<u32>().ok()?;
        let file_id = parts[2].parse::<usize>().ok()?;

        Some((start_byte, final_byte, file_id))
    }

    let new_noir_span = Location {
        span: Span::inclusive(
            convert_span(&left.span.as_string).unwrap().0,
            convert_span(&right.span.as_string).unwrap().1,
        ),
        file: state.location.file,
    };

    build_span_no_id(
        "".to_string(), // The text content is for debug info, not critical here
        Some(new_noir_span),
    )
}

// TODO: array indexing - ast_index_to_vir_expr
// TODO: tuple indexing - ast_tuple_access_to_vir_expr

pub fn parse_attribute<'a>(
    annotation: &'a str,
    mut location: Location,
    function: &'a Function,
    global_constants: &'a BTreeMap<GlobalId, (String, Type, Expression)>,
    functions: &'a BTreeMap<FuncId, Function>,
) -> Result<Attribute, Vec<ResolverError>> {
    // NOTE: #['...]
    //       ^^^^^^^ - received `Location`
    //          ^^^  - relevant stuff
    // TODO: don't do this here
    location = Location {
        span: Span::inclusive(location.span.start() + 3, location.span.end() - 1),
        ..location
    };
    let state = State {
        full_length: annotation.len() as u32,
        location,
        function,
        global_constants,
        functions,
    };
    let input = Input(annotation, &state);
    let (input, attribute_type) =
        parse_identifier(input).finish().map_err(|e| e.resolver_errors)?;

    let get_expr = || {
        let (rest, expr) = parse_expression_expr(input).finish().map_err(|e| e.resolver_errors)?;
        assert_eq!(rest.0, "");
        Ok::<_, Vec<ResolverError>>(expr)
    };

    Ok(match attribute_type {
        "ghost" => Attribute::Ghost,
        "ensures" => Attribute::Ensures(get_expr()?),
        "requires" => Attribute::Requires(get_expr()?),
        _ => {
            return Err(vec![ResolverError::VariableNotDeclared {
                name: attribute_type.to_string(),
                location,
            }]);
        }
    })
}

fn parse_bool<'a, 'b>(Input(input, state): Input<'a, 'b>) -> PResult<'a, 'b, bool> {
    alt((map(tag("true"), |_| true), map(tag("false"), |_| false))).parse(Input(input, state))
}

fn parse_sign<'a, 'b>(Input(input, state): Input<'a, 'b>) -> PResult<'a, 'b, bool> {
    let (input, opt_sign) = opt(alt((
        //
        value(false, tag(&b"-"[..])),
        value(true, tag(&b"+"[..])),
    )))
    .parse(Input(input, state))?;
    let sign = opt_sign.unwrap_or(true);

    Ok((input, sign))
}

fn parse_int<'a, 'b>(Input(input, state): Input<'a, 'b>) -> PResult<'a, 'b, BigInt> {
    let (input, sign) = parse_sign(Input(input, state))?;
    let (input, Input(digits, _)) = digit(input)?;

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

fn parse_constant_expr<'a, 'b>(Input(input, state): Input<'a, 'b>) -> PResult<'a, 'b, Expr> {
    // eprintln!("trying constant: {}", input);
    let prev_offset = input.len();
    let (Input(input, _), (typx, exprx)) = alt((
        map(parse_bool, |b| (TypX::Bool, ExprX::Const(Constant::Bool(b)))),
        map(parse_int, |bi| {
            // TODO: Better type than `TypX::Int`
            (TypX::Int(IntRange::Int), ExprX::Const(Constant::Int(bi)))
        }),
    ))
    .parse(Input(input, state))?;

    let after_offset = input.len();

    let res = build_expr(state, prev_offset, after_offset, Arc::new(typx), exprx);

    Ok((Input(input, state), res))
}

// TODO: parse identifier expression
// TODO: parse module references `fv_std::SOMETHING`
fn parse_identifier<'a, 'b>(Input(input, state): Input<'a, 'b>) -> PResult<'a, 'b, &'a str> {
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

    let (input, Input(s, _)) = parser.parse(Input(input, state))?;

    Ok((input, s))
}

fn parse_var_expr<'a, 'b>(Input(input, state): Input<'a, 'b>) -> PResult<'a, 'b, Expr> {
    // eprintln!("trying var: {}", input);
    let prev_offset = input.len();
    let (Input(input, _), ident) = parse_identifier(Input(input, state)).map_err(|_| {
        Err::Error(Error {
            resolver_errors: vec![ResolverError::ParserError(Box::new(
                NoirParserError::with_reason(
                    // TODO: ne
                    ParserErrorReason::DocCommentDoesNotDocumentAnything,
                    Location::dummy(),
                ),
            ))],
            location: Location::dummy(),
        })
    })?;
    let after_offset = input.len();

    let (variable_ident, variable_type, variable_id) = state
        .function
        .parameters
        .iter()
        .find_map(|k| {
            (k.2 == ident).then(|| {
                (ident, ast_type_to_vir_type(&k.3), VarIdentDisambiguate::RustcId(k.0.0 as usize))
            })
        })
        .or_else(|| {
            (ident == "result").then(|| {
                (FUNC_RETURN_VAR_NAME, ast_type_to_vir_type(&state.function.return_type), VarIdentDisambiguate::AirLocal)
            })
        })
        .ok_or(Err::Error(Error {
            resolver_errors: vec![ResolverError::VariableNotDeclared {
                name: ident.to_string(),
                location: build_location(state, prev_offset, after_offset),
            }],
            location: build_location(state, prev_offset, after_offset),
        }))?;

    Ok((
        Input(input, state),
        build_expr(
            state,
            prev_offset,
            after_offset,
            // NOTE: type of variable is either found in the function's parameters
            //       or is the magic `result`, which inherits the function's return type
            variable_type,
            ExprX::Var(VarIdent(Arc::new(variable_ident.to_string()), variable_id)),
        ),
    ))
}

fn parse_fn_call_expr<'a, 'b>(Input(input, state): Input<'a, 'b>) -> PResult<'a, 'b, Expr> {
    // eprintln!("trying call: {}", input);
    let prev_offset = input.len();
    let (Input(input, _), name) = parse_identifier(Input(input, state))?;

    let (Input(input, _), params) = delimited(
        tag("("),
        separated_list0(pair(tag(","), multispace), parse_expression_expr),
        tag(")"),
    )
    .parse(Input(input, state))?;
    let after_offset = input.len();

    Ok((
        Input(input, state),
        build_expr(
            state,
            prev_offset,
            after_offset,
            state
                .functions
                .iter()
                .find_map(|(_, func)| (func.name == name).then_some(&func.return_type))
                .map(ast_type_to_vir_type)
                .ok_or(Err::Error(Error {
                    resolver_errors: vec![ResolverError::VariableNotDeclared {
                        name: name.to_string(),
                        location: build_location(state, prev_offset, after_offset),
                    }],
                    location: build_location(state, prev_offset, after_offset),
                }))?,
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

fn parse_additive_expr<'a, 'b>(Input(input, state): Input<'a, 'b>) -> PResult<'a, 'b, Expr> {
    // eprintln!("trying additive: {}", input);
    let (Input(input, _), (first, remainder)) = pair(
        parse_multiplicative_expr,
        many0(pair(
            delimited(multispace, alt((tag("+"), tag("-"))), multispace),
            parse_multiplicative_expr,
        )),
    )
    .parse(Input(input, state))?;

    Ok((
        Input(input, state),
        remainder.into_iter().fold(first, |left, (Input(op, _), right)| {
            let op_kind = match op {
                "+" => BinaryOp::Arith(ArithOp::Add, Mode::Spec),
                "-" => BinaryOp::Arith(ArithOp::Sub, Mode::Spec),
                _ => unreachable!(),
            };
            let new_span = build_span_from_exprs(state, &left, &right);
            let typx = left.typ.clone();
            let exprx = ExprX::Binary(op_kind, left, right);
            SpannedTyped::new(&new_span, &typx, exprx)
        }),
    ))
}

fn parse_multiplicative_expr<'a, 'b>(Input(input, state): Input<'a, 'b>) -> PResult<'a, 'b, Expr> {
    // eprintln!("trying multiplicative: {}", input);
    let (Input(input, _), (first, remainder)) = pair(
        parse_unary_expr,
        many0(pair(
            delimited(multispace, alt((tag("*"), tag("/"), tag("%"))), multispace),
            parse_unary_expr,
        )),
    )
    .parse(Input(input, state))?;

    Ok((
        Input(input, state),
        remainder.into_iter().fold(first, |left, (Input(op, _), right)| {
            let op_kind = match op {
                "*" => BinaryOp::Arith(ArithOp::Mul, Mode::Spec),
                "/" => BinaryOp::Arith(ArithOp::EuclideanDiv, Mode::Spec),
                "%" => BinaryOp::Arith(ArithOp::EuclideanMod, Mode::Spec),
                _ => unreachable!(),
            };
            let new_span = build_span_from_exprs(state, &left, &right);
            let typx = left.typ.clone();
            let exprx = ExprX::Binary(op_kind, left, right);
            SpannedTyped::new(&new_span, &typx, exprx)
        }),
    ))
}

fn parse_expression_expr<'a, 'b>(Input(input, state): Input<'a, 'b>) -> PResult<'a, 'b, Expr> {
    // eprintln!("trying expression: {}", input);
    // NOTE: Start parsing from the lowest precedence operator
    alt((
        //
        parse_implication_expr,
    ))
    .parse(Input(input, state))
}

fn parse_parenthesised_expr<'a, 'b>(Input(input, state): Input<'a, 'b>) -> PResult<'a, 'b, Expr> {
    // eprintln!("trying parenthesised: {}", input);
    delimited(multispace, delimited(tag("("), parse_expression_expr, tag(")")), multispace)
        .parse(Input(input, state))
}

fn parse_primary_expr<'a, 'b>(Input(input, state): Input<'a, 'b>) -> PResult<'a, 'b, Expr> {
    // eprintln!("trying primary: {}", input);
    alt((
        //
        parse_parenthesised_expr,
        parse_fn_call_expr,
        parse_constant_expr,
        parse_var_expr,
    ))
    .parse(Input(input, state))
}

fn parse_unary_expr<'a, 'b>(Input(input, state): Input<'a, 'b>) -> PResult<'a, 'b, Expr> {
    // eprintln!("trying unary: {}", input);
    alt((
        map(preceded(terminated(tag("!"), multispace), parse_unary_expr), |expr| {
            let new_span = expr.span.clone();
            let typx = Arc::new(TypX::Bool);
            let exprx = ExprX::Unary(UnaryOp::Not, expr);
            SpannedTyped::new(&new_span, &typx, exprx)
        }),
        parse_primary_expr,
    ))
    .parse(Input(input, state))
}

fn parse_comparison_expr<'a, 'b>(Input(input, state): Input<'a, 'b>) -> PResult<'a, 'b, Expr> {
    // eprintln!("trying comparison: {}", input);
    let (Input(input, _), mut left) = parse_additive_expr(Input(input, state))?;

    // Comparison operators are not associative (e.g., `a < b < c` is invalid),
    // so we just look for one optional occurrence.
    if let Ok((Input(input_after, _), (Input(op, _), right))) = pair(
        delimited(
            multispace,
            alt((tag("=="), tag("!="), tag("<="), tag(">="), tag("<"), tag(">"))),
            multispace,
        ),
        parse_additive_expr,
    )
    .parse(Input(input, state))
    {
        let op_kind = match op {
            "==" => BinaryOp::Eq(Mode::Spec),
            "!=" => BinaryOp::Ne,
            "<" => BinaryOp::Inequality(vir::ast::InequalityOp::Lt),
            "<=" => BinaryOp::Inequality(vir::ast::InequalityOp::Le),
            ">" => BinaryOp::Inequality(vir::ast::InequalityOp::Gt),
            ">=" => BinaryOp::Inequality(vir::ast::InequalityOp::Ge),
            _ => unreachable!(),
        };
        let new_span = build_span_from_exprs(state, &left, &right);
        let typx = Arc::new(TypX::Bool);
        let exprx = ExprX::Binary(op_kind, left, right);
        left = SpannedTyped::new(&new_span, &typx, exprx);
        return Ok((Input(input_after, state), left));
    }

    Ok((Input(input, state), left))
}

fn parse_logical_and_expr<'a, 'b>(Input(input, state): Input<'a, 'b>) -> PResult<'a, 'b, Expr> {
    // eprintln!("trying and: {}", input);
    let (Input(input, _), (first, remainder)) = pair(
        parse_comparison_expr,
        many0(pair(delimited(multispace, tag("&"), multispace), parse_comparison_expr)),
    )
    .parse(Input(input, state))?;

    Ok((
        Input(input, state),
        remainder.into_iter().fold(first, |left, (_op, right)| {
            let new_span = build_span_from_exprs(state, &left, &right);
            let typx = Arc::new(TypX::Bool);
            let exprx = ExprX::Binary(BinaryOp::And, left, right);
            SpannedTyped::new(&new_span, &typx, exprx)
        }),
    ))
}

fn parse_logical_or_expr<'a, 'b>(Input(input, state): Input<'a, 'b>) -> PResult<'a, 'b, Expr> {
    // eprintln!("trying or: {}", input);
    let (Input(input, _), (first, remainder)) = pair(
        parse_logical_and_expr,
        many0(pair(delimited(multispace, tag("|"), multispace), parse_logical_and_expr)),
    )
    .parse(Input(input, state))?;

    Ok((
        Input(input, state),
        remainder.into_iter().fold(first, |left, (_op, right)| {
            let new_span = build_span_from_exprs(state, &left, &right);
            let typx = Arc::new(TypX::Bool);
            let exprx = ExprX::Binary(BinaryOp::Or, left, right);
            SpannedTyped::new(&new_span, &typx, exprx)
        }),
    ))
}

fn parse_implication_expr<'a, 'b>(Input(input, state): Input<'a, 'b>) -> PResult<'a, 'b, Expr> {
    // eprintln!("trying implication: {}", input);
    let (Input(input, _), (first, remainder)) = pair(
        parse_logical_or_expr,
        many0(pair(delimited(multispace, tag("==>"), multispace), parse_logical_or_expr)),
    )
    .parse(Input(input, state))?;

    Ok((
        Input(input, state),
        remainder.into_iter().fold(first, |left, (_op, right)| {
            let new_span = build_span_from_exprs(state, &left, &right);
            let typx = Arc::new(TypX::Bool);
            let exprx = ExprX::Binary(BinaryOp::Implies, left, right);
            SpannedTyped::new(&new_span, &typx, exprx)
        }),
    ))
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
            location: Location { span: Span::inclusive(1000, 2000), file: Default::default() },
            function: Box::leak(Box::new(Function {
                id: FuncId(4321),
                name: "tutmanik".to_string(),
                parameters: vec![
                    (LocalId(0), false, "a".to_string(), Type::Bool, Visibility::Public),
                    (LocalId(1), false, "kek".to_string(), Type::Unit, Visibility::Public),
                    (LocalId(2), false, "Banica_123_".to_string(), Type::Bool, Visibility::Public),
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

    fn parse<'a, 'b>(input: &'a str) -> PResult<'a, 'static, Expr> {
        parse_expression_expr(Input(input, Box::leak(Box::new(empty_state(input.len() as u32)))))
    }

    #[test]
    fn test_bool_true() {
        let expr = parse("true").unwrap();
        assert_eq!(expr.0.0, "");
        assert!(matches!(*expr.1.typ, TypX::Bool));
        assert!(matches!(expr.1.x, ExprX::Const(Constant::Bool(true))));
    }

    #[test]
    fn test_int() {
        let chislo = "1234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890";
        let expr = parse(chislo).unwrap();
        assert_eq!(expr.0.0, "");
        assert!(matches!(*expr.1.typ, TypX::Int(IntRange::Int)));
        let ExprX::Const(Constant::Int(ref bi)) = expr.1.x else { panic!() };
        assert_eq!(bi.to_str_radix(10), chislo);
    }

    #[test]
    fn test_ident() {
        let identche = "Banica_123_";
        let expr = parse(identche).unwrap();
        assert_eq!(expr.0.0, "");
        // TODO: same as definition
        assert!(matches!(*expr.1.typ, TypX::Bool));
        let ExprX::Var(VarIdent(ref i, _)) = expr.1.x else { panic!() };
        assert_eq!(**i, identche.to_string());
    }

    // #[test]
    // #[should_panic]
    // fn test_ident_starts_with_digit() {
    //     let identche = "1Banica_123_";
    //     let expr = parse_var_expr(&empty_state(12), identche).unwrap();
    //     assert_eq!(expr.0.0, "");
    //     dbg!(expr);
    // }

    #[test]
    fn test_function_call() {
        let expr = parse("banica(1, banica(a, kek))").unwrap();
        assert_eq!(expr.0.0, "");
        dbg!(expr);
    }

    #[test]
    fn test_sum() {
        let identche = "1 + 2 * 3";
        let expr = parse(identche).unwrap();
        assert_eq!(expr.0.0, "");
        dbg!(expr);
    }

    #[test]
    fn test_ghost() {
        let annotation = "ghost";
        let state = empty_state(annotation.len() as u32);
        let kek = parse_attribute(
            annotation,
            state.location,
            state.function,
            state.global_constants,
            state.functions,
        )
        .unwrap();
    }

    #[test]
    fn test_ensures() {
        let annotation = "ensures(kek < 127)";
        let state = empty_state(annotation.len() as u32);
        let kek = parse_attribute(
            annotation,
            state.location,
            state.function,
            state.global_constants,
            state.functions,
        )
        .unwrap();

        // dbg!(kek);
        panic!("{:?}", kek);
    }
}
