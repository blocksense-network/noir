use std::sync::Arc;

use crate::vir::vir_gen::{
    build_span, build_span_no_id,
    expr_to_vir::{
        expression_location,
        types::{ast_type_to_vir_type, build_tuple_type, get_bit_not_bitwidth, make_unit_vir_type},
    },
};
use noirc_errors::Location;
use noirc_frontend::{
    ast::{BinaryOpKind, QuantifierType, UnaryOp},
    monomorphization::ast::{
        Assign, Binary, Call, Cast, Expression, Function, Ident, If, LValue, Literal, Match, Type,
        Unary,
    },
    shared::Signedness,
    signed_field::SignedField,
};
use num_bigint::{BigInt, BigUint};
use num_traits::ToPrimitive;
use vir::{
    ast::{
        AirQuant, ArithOp, AutospecUsage, BinaryOp, BinderX, BitwiseOp, CallTarget, CallTargetKind,
        FieldOpr, Fun, FunX, InequalityOp, IntRange, PathX, Quant, Typ, UnaryOp as VirUnaryOp,
        UnaryOpr, VarBinder, VarBinderX, VariantCheck,
    },
    ast_util::{bitwidth_from_type, int_range_from_type, is_integer_type_signed},
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
        Expression::Cast(cast) => ast_cast_to_vir_expr(cast, mode),
        Expression::For(_) => todo!(),
        Expression::Loop(expression) => todo!(),
        Expression::While(_) => todo!(),
        Expression::If(if_expression) => {
            ast_if_to_vir_expr(if_expression, expression_location(expr), mode)
        }
        Expression::Match(_) => todo!(),
        Expression::Tuple(tuple_expressions) => {
            ast_tuple_to_vir_expr(tuple_expressions, expression_location(expr), mode)
        }
        Expression::ExtractTupleField(tuple_expression, tuple_index) => {
            ast_tuple_access_to_vir_expr(
                &tuple_expression,
                *tuple_index,
                expression_location(expr),
                mode,
            )
        }
        Expression::Call(call_expr) => ast_call_to_vir_expr(call_expr, mode),
        Expression::Let(let_expr) => todo!(),
        Expression::Constrain(expression, location, _) => {
            ast_constrain_to_vir_expr(&expression, Some(*location), mode)
        }
        Expression::Assign(assign) => {
            ast_assign_to_vir_expr(assign, expression_location(expr), mode)
        }
        Expression::Semi(expression) => todo!(),
        Expression::Clone(expression) => todo!(),
        Expression::Drop(expression) => todo!(),
        Expression::Break => ast_break_to_vir_expr(),
        Expression::Continue => ast_continue_to_vir_expr(),
        Expression::Quant(quantifier_type, idents, quantifier_body) => ast_quant_to_vir_expr(
            quantifier_type,
            idents,
            quantifier_body,
            expression_location(expr),
            mode,
        ),
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

fn ast_cast_to_vir_expr(cast_expr: &Cast, mode: Mode) -> Expr {
    let castee = ast_expr_to_vir_expr(&cast_expr.lhs, mode);
    let target_type = ast_type_to_vir_type(&cast_expr.r#type);
    // The following unwrap is safe because the semantic analysis of
    // the compiler should guarantee correctly typed expressions.
    let target_int_range =
        int_range_from_type(&target_type).expect("Can not cast to a non integer type");

    let exprx = ExprX::Unary(
        VirUnaryOp::Clip {
            range: target_int_range,
            truncate: false, // We are not truncating because Verus doesn't truncate casts
        },
        castee,
    );

    SpannedTyped::new(
        &build_span_no_id(
            format!("Cast {} to {}", cast_expr.lhs, cast_expr.r#type),
            Some(cast_expr.location),
        ),
        &target_type,
        exprx,
    )
}

fn ast_if_to_vir_expr(if_expr: &If, location: Option<Location>, mode: Mode) -> Expr {
    let condition = ast_expr_to_vir_expr(&if_expr.condition, mode);
    let consequence = ast_expr_to_vir_expr(&if_expr.consequence, mode);
    let alternative =
        if_expr.alternative.as_ref().map(|else_expr| ast_expr_to_vir_expr(&else_expr, mode));

    let exprx = ExprX::If(condition, consequence, alternative);

    SpannedTyped::new(
        &build_span_no_id(format!("If expression"), location),
        &ast_type_to_vir_type(&if_expr.typ),
        exprx,
    )
}

fn ast_tuple_to_vir_expr(
    tuple_expressions: &Vec<Expression>,
    location: Option<Location>,
    mode: Mode,
) -> Expr {
    let tuple_length = tuple_expressions.len();
    let vir_tuple_expressions: Vec<Expr> =
        tuple_expressions.iter().map(|tuple_expr| ast_expr_to_vir_expr(tuple_expr, mode)).collect();
    let tuple_types: Vec<Typ> =
        vir_tuple_expressions.iter().map(|tuple_expression| tuple_expression.typ.clone()).collect();

    let tuple_exprx = ExprX::Ctor(
        Dt::Tuple(tuple_length),
        Arc::new(format!("tuple%{tuple_length}")),
        Arc::new(
            vir_tuple_expressions
                .into_iter()
                .enumerate()
                .map(|(index, tuple_expr)| {
                    Arc::new(BinderX { name: Arc::new(index.to_string()), a: tuple_expr })
                })
                .collect(),
        ),
        None,
    );

    SpannedTyped::new(
        &build_span_no_id(format!("Tuple constructor expression"), location),
        &Arc::new(build_tuple_type(tuple_types)),
        tuple_exprx,
    )
}

fn ast_tuple_access_to_vir_expr(
    tuple_expr: &Expression,
    tuple_index: usize,
    location: Option<Location>,
    mode: Mode,
) -> Expr {
    let tuple_vir_expr = ast_expr_to_vir_expr(tuple_expr, mode);
    let tuple_expr_type =
        tuple_expr.return_type().expect("Tuple type should be known at compile time");
    let Type::Tuple(types) = tuple_expr_type.as_ref() else {
        unreachable!("Unexpected type for tuple field extraction {tuple_expr_type}");
    };
    let exprx = ExprX::UnaryOpr(
        UnaryOpr::Field(FieldOpr {
            datatype: Dt::Tuple(types.len()),
            variant: Arc::new(format!("tuple%{}", types.len())),
            field: Arc::new(tuple_index.to_string()),
            get_variant: false,
            check: VariantCheck::None,
        }),
        tuple_vir_expr,
    );

    SpannedTyped::new(
        &build_span_no_id(format!("Tuple {} access at {}", tuple_expr, tuple_index), location),
        &ast_type_to_vir_type(&types[tuple_index]),
        exprx,
    )
}

fn ast_call_to_vir_expr(call_expr: &Call, mode: Mode) -> Expr {
    let Expression::Ident(function_ident) = call_expr.func.as_ref() else {
        unreachable!("Expected functions to be presented with identifiers");
    };
    let arguments: Vec<Expr> =
        call_expr.arguments.iter().map(|argument| ast_expr_to_vir_expr(argument, mode)).collect();

    let exprx = ExprX::Call(
        CallTarget::Fun(
            CallTargetKind::Static,
            function_name_to_vir_fun(function_ident.name.clone()),
            Arc::new(vec![]),
            Arc::new(vec![]),
            AutospecUsage::Final, // We don't support the autospec attribute.
        ),
        Arc::new(arguments),
    );

    SpannedTyped::new(
        &build_span(
            function_ident.id.0,
            format!(
                "Function call {} with arguments {}",
                function_ident.name,
                call_expr
                    .arguments
                    .iter()
                    .map(|arg| arg.to_string())
                    .collect::<Vec<String>>()
                    .join(", ")
            ),
            Some(call_expr.location),
        ),
        &ast_type_to_vir_type(&call_expr.return_type),
        exprx,
    )
}

fn ast_constrain_to_vir_expr(
    assert_body_expr: &Expression,
    location: Option<Location>,
    mode: Mode,
) -> Expr {
    let exprx = ExprX::AssertAssume {
        is_assume: false,
        expr: ast_expr_to_vir_expr(assert_body_expr, mode),
    };

    SpannedTyped::new(
        &build_span_no_id(format!("Assert({})", assert_body_expr), location),
        &make_unit_vir_type(),
        exprx,
    )
}

fn ast_assign_to_vir_expr(assign_expr: &Assign, location: Option<Location>, mode: Mode) -> Expr {
    let is_lvalue_mut = is_lvalue_mut(&assign_expr.lvalue);
    let lhs_expr = ast_lvalue_to_vir_expr(&assign_expr.lvalue, location, mode);
    let exprx = ExprX::Assign {
        init_not_mut: !is_lvalue_mut,
        lhs: lhs_expr,
        rhs: ast_expr_to_vir_expr(&assign_expr.expression, mode),
        op: None,
    };

    SpannedTyped::new(
        &build_span_no_id(format!("Assign operation"), location),
        &make_unit_vir_type(),
        exprx,
    )
}

fn ast_break_to_vir_expr() -> Expr {
    SpannedTyped::new(
        &build_span_no_id(format!("Break"), None),
        &make_unit_vir_type(),
        ExprX::BreakOrContinue { label: None, is_break: true },
    )
}

fn ast_continue_to_vir_expr() -> Expr {
    SpannedTyped::new(
        &build_span_no_id(format!("Continue"), None),
        &make_unit_vir_type(),
        ExprX::BreakOrContinue { label: None, is_break: false },
    )
}

fn ast_quant_to_vir_expr(
    quantifier_type: &QuantifierType,
    quantifier_indexes: &Vec<Ident>,
    quantifier_body: &Expression,
    quantifier_location: Option<Location>,
    mode: Mode,
) -> Expr {
    let quantifier_vir_type = match quantifier_type {
        QuantifierType::Forall => Quant { quant: AirQuant::Forall },
        QuantifierType::Exists => Quant { quant: AirQuant::Exists },
    };

    let quantifier_vir_indexes: Vec<VarBinder<Typ>> = quantifier_indexes
        .iter()
        .map(|index_ident| {
            Arc::new(VarBinderX {
                name: VarIdent(
                    Arc::new(index_ident.name.clone()),
                    VarIdentDisambiguate::RustcId(index_ident.id.0.to_usize().unwrap_or(0)),
                ),
                a: if let Type::Integer(Signedness::Unsigned, _) = index_ident.typ {
                    Arc::new(TypX::Int(IntRange::Nat))
                } else {
                    Arc::new(TypX::Int(IntRange::Int))
                },
            })
        })
        .collect();

    let quantifier_vir_body = SpannedTyped::new(
        &build_span_no_id(
            format!("Quantifier {} body", quantifier_type),
            expression_location(quantifier_body),
        ),
        &Arc::new(TypX::Bool), // All quantifier bodies must be of type bool
        ast_expr_to_vir_expr(quantifier_body, mode).x.clone(),
    );

    let quantifier_vir_exprx =
        ExprX::Quant(quantifier_vir_type, Arc::new(quantifier_vir_indexes), quantifier_vir_body);

    SpannedTyped::new(
        &build_span_no_id(format!("Quantifier {}", quantifier_type), quantifier_location),
        &Arc::new(TypX::Bool),
        quantifier_vir_exprx,
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

fn function_name_to_vir_fun(func_name: String) -> Fun {
    Arc::new(FunX {
        path: Arc::new(PathX { krate: None, segments: Arc::new(vec![Arc::new(func_name)]) }),
    })
}

fn is_lvalue_mut(lvalue: &LValue) -> bool {
    match lvalue {
        LValue::Ident(ident) => ident.mutable,
        LValue::Index { array, .. } => is_lvalue_mut(&array),
        LValue::MemberAccess { object, .. } => is_lvalue_mut(&object),
        LValue::Dereference { reference, .. } => is_lvalue_mut(&reference),
    }
}

fn ast_lvalue_to_vir_expr(lvalue: &LValue, location: Option<Location>, mode: Mode) -> Expr {
    match lvalue {
        LValue::Ident(ident) => ast_ident_to_vir_expr(ident),
        LValue::Index { array, index, element_type, location } => todo!(),
        LValue::MemberAccess { object, field_index } => todo!(),
        LValue::Dereference { reference, element_type } => todo!(),
    }
}
