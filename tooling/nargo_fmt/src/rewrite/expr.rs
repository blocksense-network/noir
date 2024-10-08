use noirc_errors::Span;
use noirc_frontend::ast::{
    ArrayLiteral, BlockExpression, Expression, ExpressionKind, Literal, Path, PathKind, UnaryOp,
    UnresolvedType,
};
use noirc_frontend::token::Token;

use crate::rewrite;
use crate::visitor::{
    expr::{format_brackets, format_parens, NewlineMode},
    ExpressionType, FmtVisitor, Indent, Shape,
};

pub(crate) fn rewrite_sub_expr(
    visitor: &FmtVisitor,
    shape: Shape,
    expression: Expression,
) -> String {
    rewrite(visitor, expression, ExpressionType::SubExpression, shape)
}

pub(crate) fn rewrite(
    visitor: &FmtVisitor,
    Expression { kind, span }: Expression,
    expr_type: ExpressionType,
    shape: Shape,
) -> String {
    match kind {
        ExpressionKind::Block(block) => rewrite_block(visitor, block, span),
        ExpressionKind::Prefix(prefix) => {
            let op = match prefix.operator {
                UnaryOp::Minus => "-",
                UnaryOp::Not => "!",
                UnaryOp::MutableReference => "&mut ",
                UnaryOp::Dereference { implicitly_added } => {
                    if implicitly_added {
                        ""
                    } else {
                        "*"
                    }
                }
            };

            format!("{op}{}", rewrite_sub_expr(visitor, shape, prefix.rhs))
        }
        ExpressionKind::Cast(cast) => {
            format!("{} as {}", rewrite_sub_expr(visitor, shape, cast.lhs), cast.r#type)
        }
        kind @ ExpressionKind::Infix(_) => {
            super::infix(visitor.fork(), Expression { kind, span }, shape)
        }
        ExpressionKind::Call(call_expr) => {
            let args_span =
                visitor.span_before(call_expr.func.span.end()..span.end(), Token::LeftParen);

            let callee = rewrite_sub_expr(visitor, shape, *call_expr.func);
            let args = format_parens(
                visitor.config.fn_call_width.into(),
                visitor.fork(),
                shape,
                false,
                call_expr.arguments,
                args_span,
                true,
                NewlineMode::IfContainsNewLineAndWidth,
            );

            let bang = if call_expr.is_macro_call { "!" } else { "" };
            format!("{callee}{bang}{args}")
        }
        ExpressionKind::MethodCall(method_call_expr) => {
            let args_span = visitor.span_before(
                method_call_expr.method_name.span().end()..span.end(),
                Token::LeftParen,
            );

            let object = rewrite_sub_expr(visitor, shape, method_call_expr.object);
            let method = method_call_expr.method_name.to_string();
            let turbofish = rewrite_turbofish(visitor, shape, method_call_expr.generics);
            let args = format_parens(
                visitor.config.fn_call_width.into(),
                visitor.fork(),
                shape,
                false,
                method_call_expr.arguments,
                args_span,
                true,
                NewlineMode::IfContainsNewLineAndWidth,
            );

            let bang = if method_call_expr.is_macro_call { "!" } else { "" };
            format!("{object}.{method}{turbofish}{bang}{args}")
        }
        ExpressionKind::MemberAccess(member_access_expr) => {
            let lhs_str = rewrite_sub_expr(visitor, shape, member_access_expr.lhs);
            format!("{}.{}", lhs_str, member_access_expr.rhs)
        }
        ExpressionKind::Index(index_expr) => {
            let index_span = visitor
                .span_before(index_expr.collection.span.end()..span.end(), Token::LeftBracket);

            let collection = rewrite_sub_expr(visitor, shape, index_expr.collection);
            let index = format_brackets(visitor.fork(), false, vec![index_expr.index], index_span);

            format!("{collection}{index}")
        }
        ExpressionKind::Tuple(exprs) => format_parens(
            None,
            visitor.fork(),
            shape,
            exprs.len() == 1,
            exprs,
            span,
            true,
            NewlineMode::Normal,
        ),
        ExpressionKind::Literal(literal) => match literal {
            Literal::Integer(_, _)
            | Literal::Bool(_)
            | Literal::Str(_)
            | Literal::RawStr(..)
            | Literal::FmtStr(_) => visitor.slice(span).to_string(),
            Literal::Array(ArrayLiteral::Repeated { repeated_element, length }) => {
                let repeated = rewrite_sub_expr(visitor, shape, *repeated_element);
                let length = rewrite_sub_expr(visitor, shape, *length);

                format!("[{repeated}; {length}]")
            }
            Literal::Array(ArrayLiteral::Standard(exprs)) => {
                super::array(visitor.fork(), exprs, span, false)
            }
            Literal::Slice(ArrayLiteral::Repeated { repeated_element, length }) => {
                let repeated = rewrite_sub_expr(visitor, shape, *repeated_element);
                let length = rewrite_sub_expr(visitor, shape, *length);

                format!("&[{repeated}; {length}]")
            }
            Literal::Slice(ArrayLiteral::Standard(exprs)) => {
                super::array(visitor.fork(), exprs, span, true)
            }
            Literal::Unit => "()".to_string(),
        },
        ExpressionKind::Parenthesized(sub_expr) => {
            super::parenthesized(visitor, shape, span, *sub_expr)
        }
        ExpressionKind::Constructor(constructor) => {
            let type_name = visitor.slice(span.start()..constructor.typ.span.end());
            let fields_span =
                visitor.span_before(constructor.typ.span.end()..span.end(), Token::LeftBrace);

            visitor.format_struct_lit(type_name, fields_span, *constructor)
        }
        ExpressionKind::If(if_expr) => {
            let allow_single_line = expr_type == ExpressionType::SubExpression;

            if allow_single_line {
                let mut visitor = visitor.fork();
                visitor.indent = Indent::default();
                if let Some(line) = visitor.format_if_single_line(*if_expr.clone()) {
                    return line;
                }
            }

            visitor.format_if(*if_expr)
        }
        ExpressionKind::Variable(path) => rewrite_path(visitor, shape, path),
        ExpressionKind::Lambda(_) => visitor.slice(span).to_string(),
        ExpressionKind::Quote(_) => visitor.slice(span).to_string(),
        ExpressionKind::Comptime(block, block_span) => {
            format!("comptime {}", rewrite_block(visitor, block, block_span))
        }
        ExpressionKind::Unsafe(block, block_span) => {
            format!("unsafe {}", rewrite_block(visitor, block, block_span))
        }
        ExpressionKind::Error => unreachable!(),
        ExpressionKind::Resolved(_) => {
            unreachable!("ExpressionKind::Resolved should only emitted by the comptime interpreter")
        }
        ExpressionKind::Interned(_) => {
            unreachable!("ExpressionKind::Interned should only emitted by the comptime interpreter")
        }
        ExpressionKind::InternedStatement(_) => {
            unreachable!(
                "ExpressionKind::InternedStatement should only emitted by the comptime interpreter"
            )
        }
        ExpressionKind::Unquote(expr) => {
            if matches!(&expr.kind, ExpressionKind::Variable(..)) {
                format!("${expr}")
            } else {
                format!("$({})", rewrite_sub_expr(visitor, shape, *expr))
            }
        }
        ExpressionKind::AsTraitPath(path) => {
            let trait_path = rewrite_path(visitor, shape, path.trait_path);

            if path.trait_generics.is_empty() {
                format!("<{} as {}>::{}", path.typ, trait_path, path.impl_item)
            } else {
                let generics = path.trait_generics;
                format!("<{} as {}::{}>::{}", path.typ, trait_path, generics, path.impl_item)
            }
        }
        ExpressionKind::TypePath(path) => {
            if path.turbofish.is_empty() {
                format!("{}::{}", path.typ, path.item)
            } else {
                format!("{}::{}::{}", path.typ, path.item, path.turbofish)
            }
        }
    }
}

fn rewrite_block(visitor: &FmtVisitor, block: BlockExpression, span: Span) -> String {
    let mut visitor = visitor.fork();
    visitor.visit_block(block, span);
    visitor.finish()
}

fn rewrite_path(visitor: &FmtVisitor, shape: Shape, path: Path) -> String {
    let mut string = String::new();

    if path.kind != PathKind::Plain {
        string.push_str(&path.kind.to_string());
        string.push_str("::");
    }

    for (index, segment) in path.segments.iter().enumerate() {
        if index > 0 {
            string.push_str("::");
        }
        string.push_str(&segment.ident.to_string());
        string.push_str(&rewrite_turbofish(visitor, shape, segment.generics.clone()));
    }

    string
}

fn rewrite_turbofish(
    visitor: &FmtVisitor,
    shape: Shape,
    generics: Option<Vec<UnresolvedType>>,
) -> String {
    if let Some(generics) = generics {
        let mut turbofish = "".to_owned();
        for (i, generic) in generics.into_iter().enumerate() {
            let generic = rewrite::typ(visitor, shape, generic);
            turbofish = if i == 0 {
                format!("::<{}", generic)
            } else {
                format!("{turbofish}, {}", generic)
            };
        }
        format!("{turbofish}>")
    } else {
        "".to_owned()
    }
}
