use std::sync::Arc;

use crate::vir::vir_gen::{
    build_span, build_span_no_id,
    expr_to_vir::{
        expression_location,
        types::{ast_type_to_vir_type, get_bit_not_bitwidth, make_unit_vir_type},
    },
};
use noirc_errors::Location;
use noirc_frontend::{
    ast::{BinaryOpKind, UnaryOp},
    monomorphization::ast::{Binary, Expression, Function, Ident, Literal, Type, Unary},
    signed_field::SignedField,
};
use num_bigint::{BigInt, BigUint};
use vir::{
    ast::{ArithOp, BinaryOp, BitwiseOp, InequalityOp, Typ, UnaryOp as VirUnaryOp},
    ast_util::{bitwidth_from_type, is_integer_type_signed},
};
use vir::{
    ast::{
        Constant, Dt, Expr, ExprX, Mode, PatternX, SpannedTyped, Stmt, StmtX, TypX, VarIdent,
        VarIdentDisambiguate,
    },
    def::Spanned,
};

pub fn func_body_to_vir_expr(function: &Function, mode: Mode) -> Expr {
    let vir_expr = ast_expr_to_vir_expr(&function.body, mode);
    // TODO(totel):
    // We are unsure if we have to implement a special logic for converting a function body to a VIR expr.
    // There is a chance that we will have to map expressions into VIR expr block of statements
    // instead of directly converting them to VIR expressions.
    vir_expr
}

pub fn ast_expr_to_vir_expr(expr: &Expression, mode: Mode) -> Expr {
    match expr {
        Expression::Ident(ident) => ast_ident_to_vir_expr(ident),
        Expression::Literal(literal) => ast_literal_to_vir_expr(literal),
        Expression::Block(expressions) => {
            ast_block_to_vir_expr(expressions, expression_location(expr), mode)
        }
        Expression::Unary(unary) => ast_unary_to_vir_expr(unary, mode),
        Expression::Binary(binary) => ast_binary_to_vir_expr(binary, mode),
        Expression::Index(index) => todo!(),
        Expression::Cast(cast) => todo!(),
        Expression::For(_) => todo!(),
        Expression::Loop(expression) => todo!(),
        Expression::While(_) => todo!(),
        Expression::If(_) => todo!(),
        Expression::Match(_) => todo!(),
        Expression::Tuple(expressions) => todo!(),
        Expression::ExtractTupleField(expression, _) => todo!(),
        Expression::Call(call) => todo!(),
        Expression::Let(let_expr) => todo!(),
        Expression::Constrain(expression, location, _) => todo!(),
        Expression::Assign(assign) => todo!(),
        Expression::Semi(expression) => todo!(),
        Expression::Clone(expression) => todo!(),
        Expression::Drop(expression) => todo!(),
        Expression::Break => todo!(),
        Expression::Continue => todo!(),
        Expression::Quant(quantifier_type, idents, expression) => todo!(),
    }
}

fn ast_ident_to_vir_expr(ident: &Ident) -> Expr {
    let exprx = ExprX::Var(VarIdent(
        Arc::new(ident.name.clone()),
        VarIdentDisambiguate::RustcId(
            ident.id.0.try_into().expect("Failed to convert var ast id to usize"),
        ),
    ));

    SpannedTyped::new(
        &build_span(ident.id.0, format!("Var {}", ident.name), ident.location),
        &ast_type_to_vir_type(&ident.typ),
        exprx,
    )
}

fn ast_literal_to_vir_expr(literal: &Literal) -> Expr {
    let expr = match literal {
        Literal::Array(array_literal) => todo!(),
        Literal::Slice(array_literal) => todo!(),
        Literal::Integer(signed_field, ast_type, location) => {
            let exprx = numeric_const_to_vir_exprx(signed_field);
            SpannedTyped::new(
                &build_span_no_id(format!("Integer literal"), Some(*location)),
                &ast_type_to_vir_type(ast_type),
                exprx,
            )
        }
        Literal::Bool(bool_value) => {
            let exprx = ExprX::Const(Constant::Bool(*bool_value));
            SpannedTyped::new(
                &build_span_no_id(format!("Boolean literal"), None),
                &ast_type_to_vir_type(&Type::Bool),
                exprx,
            )
        }
        Literal::Unit => {
            let exprx = ExprX::Ctor(
                Dt::Tuple(0),
                Arc::new(String::from("tuple%0")),
                Arc::new(Vec::new()),
                None,
            );
            SpannedTyped::new(
                &build_span_no_id(format!("Unit literal"), None),
                &ast_type_to_vir_type(&Type::Unit),
                exprx,
            )
        }
        Literal::Str(_) => todo!(),
        Literal::FmtStr(fmt_str_fragments, _, expression) => todo!(),
    };

    expr
}

fn numeric_const_to_vir_exprx(signed_field: &SignedField) -> ExprX {
    let const_big_uint: BigUint = signed_field.absolute_value().into_repr().into();
    let big_int_sign =
        if signed_field.is_negative() { num_bigint::Sign::Minus } else { num_bigint::Sign::Plus };
    let const_big_int: BigInt = BigInt::from_biguint(big_int_sign, const_big_uint.clone());

    ExprX::Const(Constant::Int(const_big_int))
}

fn ast_block_to_vir_expr(block: &Vec<Expression>, location: Option<Location>, mode: Mode) -> Expr {
    let mut stmts: Vec<Stmt> = block.iter().map(|expr| ast_expr_to_stmt(expr, mode)).collect();

    let (last_expr, block_type) = match stmts.pop() {
        Some(stmt) => match &stmt.as_ref().x {
            StmtX::Expr(expr) => {
                // We want to return the last expression separately
                let typ = expr.typ.clone();
                (Some(expr.clone()), typ)
            }
            StmtX::Decl { .. } => {
                // Put it back if it's a Decl
                stmts.push(stmt);
                (None, make_unit_vir_type())
            }
        },
        None => (None, make_unit_vir_type()),
    };

    let exprx = ExprX::Block(Arc::new(stmts), last_expr);

    SpannedTyped::new(
        &build_span_no_id(format!("Block of statements"), location),
        &block_type,
        exprx,
    )
}

fn ast_expr_to_stmt(expr: &Expression, mode: Mode) -> Stmt {
    if let Expression::Let(let_expr) = expr {
        let patternx = PatternX::Var {
            name: build_var_ident(let_expr.name.clone(), let_expr.id.0),
            mutable: let_expr.mutable,
        };
        let init_expr = ast_expr_to_vir_expr(&let_expr.expression, mode);
        let pattern = SpannedTyped::new(&init_expr.span, &init_expr.typ, patternx);
        let stmtx =
            StmtX::Decl { pattern: pattern, mode: Some(mode), init: Some(init_expr), els: None };

        Spanned::new(
            build_span(let_expr.id.0, format!("Let expression"), expression_location(expr)),
            stmtx,
        )
    } else {
        let expr_as_vir = ast_expr_to_vir_expr(expr, mode);
        let stmtx = StmtX::Expr(expr_as_vir.clone());

        Spanned::new(expr_as_vir.span.clone(), stmtx)
    }
}

fn ast_unary_to_vir_expr(unary_expr: &Unary, mode: Mode) -> Expr {
    let exprx = match (unary_expr.operator, &unary_expr.result_type) {
        (UnaryOp::Minus, _) => todo!(),
        (UnaryOp::Not, Type::Bool) => {
            ExprX::Unary(VirUnaryOp::Not, ast_expr_to_vir_expr(&unary_expr.rhs, mode))
        }
        (UnaryOp::Not, ast_type) => ExprX::Unary(
            VirUnaryOp::BitNot(get_bit_not_bitwidth(ast_type)),
            ast_expr_to_vir_expr(&unary_expr.rhs, mode),
        ),
        (UnaryOp::Reference { mutable }, ast_type) => todo!(),
        (UnaryOp::Dereference { implicitly_added }, ast_type) => todo!(),
    };

    SpannedTyped::new(
        &build_span_no_id(format!("Unary operation"), Some(unary_expr.location)),
        &ast_type_to_vir_type(&unary_expr.result_type),
        exprx,
    )
}

fn ast_binary_to_vir_expr(binary_expr: &Binary, mode: Mode) -> Expr {
    let lhs_expr = ast_expr_to_vir_expr(&binary_expr.lhs, mode);
    let rhs_expr = ast_expr_to_vir_expr(&binary_expr.rhs, mode);
    let binary_op = binary_op_to_vir_binary_op(&binary_expr.operator, mode, &lhs_expr.typ);
    let exprx = ExprX::Binary(binary_op, lhs_expr.clone(), rhs_expr);

    SpannedTyped::new(
        &build_span_no_id(
            format!("{} {} {}", binary_expr.lhs, binary_expr.operator, binary_expr.rhs),
            Some(binary_expr.location),
        ),
        &lhs_expr.typ,
        exprx,
    )
}

fn binary_op_to_vir_binary_op(
    ast_binary_op: &BinaryOpKind,
    mode: Mode,
    lhs_type: &Typ,
) -> BinaryOp {
    match ast_binary_op {
        BinaryOpKind::Add => BinaryOp::Arith(ArithOp::Add, mode),
        BinaryOpKind::Subtract => BinaryOp::Arith(ArithOp::Sub, mode),
        BinaryOpKind::Multiply => BinaryOp::Arith(ArithOp::Mul, mode),
        BinaryOpKind::Divide => BinaryOp::Arith(ArithOp::EuclideanDiv, mode),
        BinaryOpKind::Equal => BinaryOp::Eq(mode),
        BinaryOpKind::NotEqual => BinaryOp::Ne,
        BinaryOpKind::Less => BinaryOp::Inequality(InequalityOp::Lt),
        BinaryOpKind::LessEqual => BinaryOp::Inequality(InequalityOp::Le),
        BinaryOpKind::Greater => BinaryOp::Inequality(InequalityOp::Gt),
        BinaryOpKind::GreaterEqual => BinaryOp::Inequality(InequalityOp::Ge),
        BinaryOpKind::And => BinaryOp::And,
        BinaryOpKind::Or => BinaryOp::Or,
        BinaryOpKind::Xor => BinaryOp::Xor,
        BinaryOpKind::ShiftRight => BinaryOp::Bitwise(
            BitwiseOp::Shr(
                bitwidth_from_type(lhs_type).expect("Bitwise operation with non int type"),
            ),
            mode,
        ),
        BinaryOpKind::ShiftLeft => BinaryOp::Bitwise(
            BitwiseOp::Shl(
                bitwidth_from_type(lhs_type).expect("Bitwise operation with non int type"),
                is_integer_type_signed(lhs_type),
            ),
            mode,
        ),
        BinaryOpKind::Modulo => BinaryOp::Arith(ArithOp::EuclideanMod, mode),
        BinaryOpKind::Implication => BinaryOp::Implies,
    }
}

fn build_var_ident(name: String, id: u32) -> VarIdent {
    VarIdent(
        Arc::new(name.clone()),
        VarIdentDisambiguate::RustcId(
            id.try_into().expect(&format!("Failed to convert id for var {}", name)),
        ),
    )
}
