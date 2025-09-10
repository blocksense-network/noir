use std::{collections::BTreeMap, sync::Arc};

use crate::vir::vir_gen::{
    build_span, build_span_no_id,
    expr_to_vir::{
        expression_location,
        std_functions::handle_fv_std_call,
        types::{
            ast_const_to_vir_type_const, ast_type_to_vir_type, build_tuple_type,
            get_binary_op_type, get_bit_not_bitwidth, get_collection_type_len, is_inner_type_array,
            is_type_field, make_unit_vir_type,
        },
    },
};
use acvm::{AcirField, FieldElement};
use noirc_errors::Location;
use noirc_frontend::{
    ast::{BinaryOpKind, QuantifierType, UnaryOp},
    monomorphization::{
        FUNC_RETURN_VAR_NAME,
        ast::{
            Assign, Binary, Call, Cast, Definition, Expression, Function, GlobalId, Ident, If,
            Index, LValue, Literal, Match, Type, Unary, While,
        },
    },
    shared::Signedness,
    signed_field::SignedField,
};
use num_bigint::{BigInt, BigUint};
use num_traits::ToPrimitive;
use vir::{
    ast::{
        AirQuant, ArithOp, AutospecUsage, BinaryOp, BinderX, BitwiseOp, CallTarget, CallTargetKind,
        FieldOpr, Fun, FunX, Idents, ImplPath, ImplPaths, InequalityOp, IntRange, PathX, Primitive,
        Quant, Typ, TypDecoration, Typs, UnaryOp as VirUnaryOp, UnaryOpr, VarBinder, VarBinderX,
        VariantCheck,
    },
    ast_util::{bitwidth_from_type, int_range_from_type, is_integer_type_signed, unit_typ},
};
use vir::{
    ast::{
        Constant, Dt, Expr, ExprX, Mode, PatternX, SpannedTyped, Stmt, StmtX, TypX, VarIdent,
        VarIdentDisambiguate,
    },
    def::Spanned,
};

pub fn func_body_to_vir_expr(
    function: &Function,
    mode: Mode,
    globals: &BTreeMap<GlobalId, (String, Type, Expression)>,
) -> Expr {
    let vir_expr = ast_expr_to_vir_expr(&function.body, mode, globals);
    // TODO(totel):
    // We are unsure if we have to implement a special logic for converting a function body to a VIR expr.
    // There is a chance that we will have to map expressions into VIR expr block of statements
    // instead of directly converting them to VIR expressions.
    vir_expr
}

pub fn ast_expr_to_vir_expr(
    expr: &Expression,
    mode: Mode,
    globals: &BTreeMap<GlobalId, (String, Type, Expression)>,
) -> Expr {
    match expr {
        Expression::Ident(ident) => ast_ident_to_vir_expr(ident),
        Expression::Literal(literal) => {
            ast_literal_to_vir_expr(literal, expression_location(expr), mode, globals)
        }
        Expression::Block(expressions) => {
            ast_block_to_vir_expr(expressions, expression_location(expr), mode, globals)
        }
        Expression::Unary(unary) => ast_unary_to_vir_expr(unary, mode, globals),
        Expression::Binary(binary) => ast_binary_to_vir_expr(binary, mode, globals),
        Expression::Index(index) => ast_index_to_vir_expr(index, mode, globals),
        Expression::Cast(cast) => ast_cast_to_vir_expr(cast, mode, globals),
        Expression::For(_) => todo!(), //TODO(totel) This is a very complicated expression to convert
        Expression::Loop(loop_body) => {
            ast_loop_to_vir_expr(loop_body, expression_location(expr), mode, globals)
        }
        Expression::While(while_expression) => {
            ast_while_to_vir_expr(while_expression, expression_location(expr), mode, globals)
        }
        Expression::If(if_expression) => {
            ast_if_to_vir_expr(if_expression, expression_location(expr), mode, globals)
        }
        Expression::Match(_) => todo!(),
        Expression::Tuple(tuple_expressions) => {
            ast_tuple_to_vir_expr(tuple_expressions, expression_location(expr), mode, globals)
        }
        Expression::ExtractTupleField(tuple_expression, tuple_index) => {
            ast_tuple_access_to_vir_expr(
                &tuple_expression,
                *tuple_index,
                expression_location(expr),
                mode,
                globals,
            )
        }
        Expression::Call(call_expr) => ast_call_to_vir_expr(call_expr, mode, globals),
        Expression::Let(let_expr) => todo!(),
        Expression::Constrain(expression, location, _) => {
            ast_constrain_to_vir_expr(&expression, Some(*location), globals)
        }
        Expression::Assign(assign) => {
            ast_assign_to_vir_expr(assign, expression_location(expr), mode, globals)
        }
        Expression::Semi(expression) => ast_expr_to_vir_expr(&expression, mode, globals),
        Expression::Clone(expression) => {
            // The `Clone` expression is introduced into the AST during the `handle_ownership` pass.
            // These expressions are inserted only in unconstrained Noir code and conceptually represent
            // an increment to the reference count of arrays.
            //
            // In Verus, array cloning is not supported. However, Rust treats arrays as `Copy` types,
            // allowing them to be duplicated implicitly without invoking `Clone`.
            //
            // As a result, when we encounter a `Clone` expression, we simply return its inner expression,
            // effectively ignoring the cloning operation.

            ast_expr_to_vir_expr(expression, mode, globals)
        }
        Expression::Drop(expression) => {
            // Similar to `Clone`, the `Drop` expression is introduced during the `handle_ownership` pass.
            // It is only inserted in unconstrained Noir code and represents a decrement to the reference count of arrays.
            //
            // Since Verus does not support `core::mem::drop`, we consume the inner expression
            // and return a unit value instead, effectively ignoring the drop operation.

            vir::ast_util::mk_tuple(
                &build_span_no_id(format!("Drop {}", expression), expression_location(expr)),
                &Arc::new(Vec::new()),
            )
        }
        Expression::Break => ast_break_to_vir_expr(),
        Expression::Continue => ast_continue_to_vir_expr(),
        Expression::Quant(quantifier_type, idents, quantifier_body) => ast_quant_to_vir_expr(
            quantifier_type,
            idents,
            quantifier_body,
            expression_location(expr),
            mode,
            globals,
        ),
    }
}

fn ast_ident_to_vir_expr(ident: &Ident) -> Expr {
    let ident_id: u32 =
        ast_definition_to_id(&ident.definition).expect("Definition doesn't have an id");
    let var_ident = ast_ident_to_vir_var_ident(ident, ident_id);

    let exprx = if matches!(ident.definition, Definition::Global(_)) {
        ExprX::ConstVar(
            Arc::new(FunX {
                path: Arc::new(PathX {
                    krate: None,
                    segments: Arc::new(vec![Arc::new(ident.name.clone())]),
                }),
            }),
            AutospecUsage::Final,
        )
    } else {
        ExprX::Var(var_ident)
    };

    SpannedTyped::new(
        &build_span(ident_id, format!("Var {}", ident.name), ident.location),
        &ast_type_to_vir_type(&ident.typ),
        exprx,
    )
}

fn ast_literal_to_vir_expr(
    literal: &Literal,
    location: Option<Location>,
    mode: Mode,
    globals: &BTreeMap<GlobalId, (String, Type, Expression)>,
) -> Expr {
    let expr = match literal {
        Literal::Array(array_literal) => {
            let exprx = ExprX::ArrayLiteral(Arc::new(
                array_literal
                    .contents
                    .iter()
                    .map(|expr| ast_expr_to_vir_expr(expr, mode, globals))
                    .collect(),
            ));
            SpannedTyped::new(
                &build_span_no_id(format!("Array literal expression"), location),
                &ast_type_to_vir_type(&array_literal.typ),
                exprx,
            )
        }
        Literal::Slice(array_literal) => todo!(),
        Literal::Integer(signed_field, ast_type, location) => {
            let exprx = numeric_const_to_vir_exprx(signed_field, ast_type);
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

pub fn numeric_const_to_vir_exprx(signed_field: &SignedField, ast_type: &Type) -> ExprX {
    // If we have a negative Field const we want to wrap it around the finite field modulus
    let (const_big_uint, big_int_sign): (BigUint, _) = {
        match ast_type {
            Type::Field => {
                // All Fields after wrapping are positive numbers
                (signed_field.to_field_element().into_repr().into(), num_bigint::Sign::Plus)
            }
            Type::Integer(..) => match signed_field.is_negative() {
                false => (signed_field.absolute_value().into_repr().into(), num_bigint::Sign::Plus),
                true => (signed_field.absolute_value().into_repr().into(), num_bigint::Sign::Minus),
            },
            _ => unreachable!(), // Integers can be only of type Field or Integer
        }
    };
    let const_big_int: BigInt = BigInt::from_biguint(big_int_sign, const_big_uint.clone());

    ExprX::Const(Constant::Int(const_big_int))
}

fn ast_block_to_vir_expr(
    block: &Vec<Expression>,
    location: Option<Location>,
    mode: Mode,
    globals: &BTreeMap<GlobalId, (String, Type, Expression)>,
) -> Expr {
    let mut stmts: Vec<Stmt> =
        block.iter().map(|expr| ast_expr_to_stmt(expr, mode, globals)).collect();

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

fn ast_expr_to_stmt(
    expr: &Expression,
    mode: Mode,
    globals: &BTreeMap<GlobalId, (String, Type, Expression)>,
) -> Stmt {
    if let Expression::Let(let_expr) = expr {
        let patternx = PatternX::Var {
            name: build_var_ident(let_expr.name.clone(), let_expr.id.0),
            mutable: let_expr.mutable,
        };
        let init_expr = ast_expr_to_vir_expr(&let_expr.expression, mode, globals);
        let pattern = SpannedTyped::new(&init_expr.span, &init_expr.typ, patternx);
        let stmtx =
            StmtX::Decl { pattern: pattern, mode: Some(mode), init: Some(init_expr), els: None };

        Spanned::new(
            build_span(let_expr.id.0, format!("Let expression"), expression_location(expr)),
            stmtx,
        )
    } else {
        let expr_as_vir = ast_expr_to_vir_expr(expr, mode, globals);
        let stmtx = StmtX::Expr(expr_as_vir.clone());

        Spanned::new(expr_as_vir.span.clone(), stmtx)
    }
}

fn ast_unary_to_vir_expr(
    unary_expr: &Unary,
    mode: Mode,
    globals: &BTreeMap<GlobalId, (String, Type, Expression)>,
) -> Expr {
    let exprx = match (unary_expr.operator, &unary_expr.result_type) {
        (UnaryOp::Minus, _) => todo!(),
        (UnaryOp::Not, Type::Bool) => {
            ExprX::Unary(VirUnaryOp::Not, ast_expr_to_vir_expr(&unary_expr.rhs, mode, globals))
        }
        (UnaryOp::Not, ast_type) => ExprX::Unary(
            VirUnaryOp::BitNot(get_bit_not_bitwidth(ast_type)),
            ast_expr_to_vir_expr(&unary_expr.rhs, mode, globals),
        ),
        (UnaryOp::Reference { mutable }, ast_type) => {
            let expr: Expr = SpannedTyped::new(
                &build_span_no_id(
                    format!("Mutable:{} Reference {}", mutable, unary_expr.rhs),
                    Some(unary_expr.location),
                ),
                &Arc::new(TypX::Decorate(
                    if mutable { TypDecoration::MutRef } else { TypDecoration::Ref },
                    None,
                    ast_type_to_vir_type(ast_type),
                )),
                {
                    let inner_expr = if let Expression::Ident(ident) = &*unary_expr.rhs {
                        // Mutable reference to ident must be handled with ExprX::VarLoc
                        SpannedTyped::new(
                            &build_span(ident.id.0, format!("&mut {}", ident.name), ident.location),
                            &ast_type_to_vir_type(&ident.typ),
                            ExprX::VarLoc(ast_ident_to_vir_var_ident(
                                &ident,
                                ast_definition_to_id(&ident.definition).unwrap(),
                            )),
                        )
                    } else {
                        ast_expr_to_vir_expr(&unary_expr.rhs, mode, globals)
                    };
                    if mutable {
                        ExprX::Loc(inner_expr)
                    } else {
                        inner_expr.x.clone() // Can not move out of Arc
                    }
                },
            );
            return expr;
        }
        (UnaryOp::Dereference { .. }, ast_type) => {
            // If we have Dereference(Expr), Verus treats them as if it is only Expr.
            // Therefore we will do the same and ignore the Dereference operation.
            let expr = SpannedTyped::new(
                &build_span_no_id(
                    format!("Dereference {}", unary_expr.rhs),
                    Some(unary_expr.location),
                ),
                &ast_type_to_vir_type(ast_type),
                ast_expr_to_vir_expr(&unary_expr.rhs, mode, globals).x.clone(), // Can not move out of Arc
            );

            return expr;
        }
    };

    SpannedTyped::new(
        &build_span_no_id(format!("Unary operation"), Some(unary_expr.location)),
        &ast_type_to_vir_type(&unary_expr.result_type),
        exprx,
    )
}

fn ast_binary_to_vir_expr(
    binary_expr: &Binary,
    mode: Mode,
    globals: &BTreeMap<GlobalId, (String, Type, Expression)>,
) -> Expr {
    let lhs_expr = ast_expr_to_vir_expr(&binary_expr.lhs, mode, globals);
    let rhs_expr = ast_expr_to_vir_expr(&binary_expr.rhs, mode, globals);
    let binary_op = binary_op_to_vir_binary_op(&binary_expr.operator, mode, &lhs_expr.typ);
    let binary_op_type = get_binary_op_type(lhs_expr.typ.clone(), &binary_expr.operator);

    let exprx = ExprX::Binary(binary_op, lhs_expr, rhs_expr);

    let vir_binary_expr = SpannedTyped::new(
        &build_span_no_id(
            format!("{} {} {}", binary_expr.lhs, binary_expr.operator, binary_expr.rhs),
            Some(binary_expr.location),
        ),
        &binary_op_type,
        exprx,
    );

    if is_type_field(
        binary_expr.lhs.return_type().expect("Lhs of binary expression must have a type").as_ref(),
    ) && is_type_field(
        binary_expr.rhs.return_type().expect("Rhs of binary expression must have a type").as_ref(),
    ) && binary_expr.operator.is_arithmetic()
    {
        wrap_with_field_modulo(vir_binary_expr, mode)
    } else {
        vir_binary_expr
    }
}

fn ast_cast_to_vir_expr(
    cast_expr: &Cast,
    mode: Mode,
    globals: &BTreeMap<GlobalId, (String, Type, Expression)>,
) -> Expr {
    let castee = ast_expr_to_vir_expr(&cast_expr.lhs, mode, globals);
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

fn ast_if_to_vir_expr(
    if_expr: &If,
    location: Option<Location>,
    mode: Mode,
    globals: &BTreeMap<GlobalId, (String, Type, Expression)>,
) -> Expr {
    let condition = ast_expr_to_vir_expr(&if_expr.condition, mode, globals);
    let consequence = ast_expr_to_vir_expr(&if_expr.consequence, mode, globals);
    let alternative = if_expr
        .alternative
        .as_ref()
        .map(|else_expr| ast_expr_to_vir_expr(&else_expr, mode, globals));

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
    globals: &BTreeMap<GlobalId, (String, Type, Expression)>,
) -> Expr {
    let tuple_length = tuple_expressions.len();
    let vir_tuple_expressions: Vec<Expr> = tuple_expressions
        .iter()
        .map(|tuple_expr| ast_expr_to_vir_expr(tuple_expr, mode, globals))
        .collect();
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
    globals: &BTreeMap<GlobalId, (String, Type, Expression)>,
) -> Expr {
    let tuple_vir_expr = ast_expr_to_vir_expr(tuple_expr, mode, globals);
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

fn ast_call_to_vir_expr(
    call_expr: &Call,
    mode: Mode,
    globals: &BTreeMap<GlobalId, (String, Type, Expression)>,
) -> Expr {
    // Handle function calls to `fv_std_lib`
    if let Some(expr) = handle_fv_std_call(call_expr, mode, globals) {
        return expr;
    }

    let Expression::Ident(function_ident) = call_expr.func.as_ref() else {
        unreachable!("Expected functions to be presented with identifiers");
    };
    let arguments: Vec<Expr> = call_expr
        .arguments
        .iter()
        .map(|argument| ast_expr_to_vir_expr(argument, mode, globals))
        .collect();

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
            ast_definition_to_id(&function_ident.definition).expect("Functions must have an id"),
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
    globals: &BTreeMap<GlobalId, (String, Type, Expression)>,
) -> Expr {
    let exprx = ExprX::AssertAssume {
        is_assume: false,
        expr: ast_expr_to_vir_expr(assert_body_expr, Mode::Spec, globals),
    };

    let assert_expr = SpannedTyped::new(
        &build_span_no_id(format!("Assert({})", assert_body_expr), location),
        &make_unit_vir_type(),
        exprx,
    );

    wrap_with_ghost_block(assert_expr, location)
}

pub fn wrap_with_ghost_block(expr: Expr, location: Option<Location>) -> Expr {
    let block_wrap = SpannedTyped::new(
        &build_span_no_id("A wrapper block for assert expression".to_string(), location),
        &make_unit_vir_type(),
        ExprX::Block(Arc::new(Vec::new()), Some(expr)),
    );

    SpannedTyped::new(
        &build_span_no_id("Ghost wrapper for AssertAssume".to_string(), location),
        &make_unit_vir_type(),
        ExprX::Ghost { alloc_wrapper: false, tracked: false, expr: block_wrap },
    )
}

fn ast_assign_to_vir_expr(
    assign_expr: &Assign,
    location: Option<Location>,
    mode: Mode,
    globals: &BTreeMap<GlobalId, (String, Type, Expression)>,
) -> Expr {
    if let LValue::Index { array, index, element_type, location } = &assign_expr.lvalue {
        return ast_array_assign_to_vir_expr(
            &array,
            &index,
            &element_type,
            &assign_expr.expression,
            Some(*location),
            mode,
            globals,
        );
    }
    let is_lvalue_mut = is_lvalue_mut(&assign_expr.lvalue)
        || matches!(get_lvalue_ident(&assign_expr.lvalue).typ, Type::Reference(_, true));
    let lhs_expr = ast_lvalue_to_vir_expr(&assign_expr.lvalue, location, mode);
    let exprx = ExprX::Assign {
        init_not_mut: !is_lvalue_mut,
        lhs: lhs_expr,
        rhs: ast_expr_to_vir_expr(&assign_expr.expression, mode, globals),
        op: None,
    };

    SpannedTyped::new(
        &build_span_no_id(format!("Assign operation"), location),
        &make_unit_vir_type(),
        exprx,
    )
}

fn ast_array_assign_to_vir_expr(
    array: &LValue,
    index: &Expression,
    element_type: &Type,
    rhs_to_be_assigned_expr: &Expression,
    location: Option<Location>,
    mode: Mode,
    globals: &BTreeMap<GlobalId, (String, Type, Expression)>,
) -> Expr {
    let array_ident = get_lvalue_ident(array);
    let lhs_array_expr = SpannedTyped::new(
        &build_span_no_id(
            format!("Lhs array identifier {}", array_ident.name),
            array_ident.location,
        ),
        &ast_type_to_vir_type(&array_ident.typ),
        ExprX::Loc(ast_lvalue_ident_to_vir_expr(array_ident)),
    );

    let exprx = ExprX::Call(
        CallTarget::Fun(
            CallTargetKind::Static,
            Arc::new(FunX {
                path: Arc::new(PathX {
                    krate: Some(Arc::new("vstd".to_string())),
                    segments: Arc::new(vec![
                        Arc::new("std_specs".to_string()),
                        Arc::new("core".to_string()),
                        Arc::new("index_set".to_string()),
                    ]),
                }),
            }),
            Arc::new(vec![
                ast_type_to_vir_type(&array_ident.typ),
                Arc::new(TypX::Int(IntRange::USize)), // Type of the index
                ast_type_to_vir_type(element_type),
            ]),
            Arc::new(Vec::new()),
            AutospecUsage::Final,
        ),
        Arc::new(vec![
            lhs_array_expr,
            ast_expr_to_vir_expr(index, mode, globals),
            ast_expr_to_vir_expr(rhs_to_be_assigned_expr, mode, globals),
        ]),
    );

    SpannedTyped::new(
        &build_span_no_id(
            format!("{} at index {} = {}", array_ident.name, index, rhs_to_be_assigned_expr),
            location,
        ),
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
    globals: &BTreeMap<GlobalId, (String, Type, Expression)>,
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
                    VarIdentDisambiguate::RustcId(
                        ast_definition_to_id(&index_ident.definition)
                            .expect("Quantifier indexes should have a local id")
                            .try_into()
                            .expect("Failed to convert u32 id to usize"),
                    ),
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
        ast_expr_to_vir_expr(quantifier_body, mode, globals).x.clone(),
    );

    let quantifier_vir_exprx =
        ExprX::Quant(quantifier_vir_type, Arc::new(quantifier_vir_indexes), quantifier_vir_body);

    SpannedTyped::new(
        &build_span_no_id(format!("Quantifier {}", quantifier_type), quantifier_location),
        &Arc::new(TypX::Bool),
        quantifier_vir_exprx,
    )
}

fn ast_index_to_vir_expr(
    index: &Index,
    mode: Mode,
    globals: &BTreeMap<GlobalId, (String, Type, Expression)>,
) -> Expr {
    let array_expr = ast_expr_to_vir_expr(&index.collection, mode, globals);
    let index_expr = ast_expr_to_vir_expr(&index.index, mode, globals);
    let element_type = ast_type_to_vir_type(&index.element_type);

    let array_type = index.collection.return_type();
    assert!(
        array_type.as_ref().map_or(false, |inner_type| is_inner_type_array(inner_type.as_ref())),
        "Found a type which can not be indexed"
    );
    let array_len = get_collection_type_len(array_type.unwrap().as_ref())
        .expect("Collections must have a length");
    let array_len_as_type =
        ast_const_to_vir_type_const(array_len.try_into().expect("Failed to convert u32 to usize"));
    let array_inner_type_and_length_type: Typs =
        Arc::new(vec![element_type.clone(), array_len_as_type.clone()]);

    let vstd_krate = Some(Arc::new("vstd".to_string()));

    let segments: Idents;
    let call_target_kind: CallTargetKind;
    let typs_for_vstd_func_call: Typs;
    let trait_impl_paths: ImplPaths;
    let autospec_usage: AutospecUsage;
    match mode {
        Mode::Spec | Mode::Proof => {
            segments = Arc::new(vec![
                Arc::new("array".to_string()),
                Arc::new("ArrayAdditionalSpecFns".to_string()),
                Arc::new("spec_index".to_string()),
            ]);
            let segments_for_resolved = Arc::new(vec![
                Arc::new("array".to_string()),
                Arc::new("impl&%2".to_string()),
                Arc::new("spec_index".to_string()),
            ]);
            call_target_kind = CallTargetKind::DynamicResolved {
                resolved: Arc::new(FunX {
                    path: Arc::new(PathX {
                        krate: vstd_krate.clone(),
                        segments: segments_for_resolved,
                    }),
                }),
                typs: array_inner_type_and_length_type,
                impl_paths: Arc::new(vec![]),
                is_trait_default: false,
            };
            let array_as_primary_vir_type = Arc::new(TypX::Primitive(
                Primitive::Array,
                Arc::new(vec![element_type.clone(), array_len_as_type.clone()]),
            ));
            typs_for_vstd_func_call =
                Arc::new(vec![array_as_primary_vir_type, element_type.clone()]);
            let trait_impl_path1 = ImplPath::TraitImplPath(Arc::new(PathX {
                krate: vstd_krate.clone(),
                segments: Arc::new(vec![
                    Arc::new("array".to_string()),
                    Arc::new("impl&%0".to_string()),
                ]),
            }));
            let trait_impl_path2 = ImplPath::TraitImplPath(Arc::new(PathX {
                krate: vstd_krate.clone(),
                segments: Arc::new(vec![
                    Arc::new("array".to_string()),
                    Arc::new("impl&%2".to_string()),
                ]),
            }));
            trait_impl_paths = Arc::new(vec![trait_impl_path1, trait_impl_path2]);
            autospec_usage = AutospecUsage::IfMarked;
        }
        Mode::Exec => {
            segments = Arc::new(vec![
                Arc::new("array".to_string()),
                Arc::new("array_index_get".to_string()),
            ]);
            call_target_kind = CallTargetKind::Static;
            let trait_impl_path = ImplPath::TraitImplPath(Arc::new(PathX {
                krate: None,
                segments: Arc::new(vec![Arc::new("fun%0".to_string())]),
            }));
            typs_for_vstd_func_call = array_inner_type_and_length_type;
            trait_impl_paths = Arc::new(vec![trait_impl_path]);
            autospec_usage = AutospecUsage::Final;
        }
    };

    let array_get_vir_exprx: ExprX = ExprX::Call(
        CallTarget::Fun(
            call_target_kind.clone(),
            Arc::new(FunX {
                path: Arc::new(PathX { krate: vstd_krate.clone(), segments: segments.clone() }),
            }),
            typs_for_vstd_func_call.clone(),
            trait_impl_paths.clone(),
            autospec_usage,
        ),
        Arc::new(vec![array_expr.clone(), index_expr]),
    );

    SpannedTyped::new(
        &build_span_no_id(
            format!("Array {}, indexed by {}", index.collection, index.index),
            Some(index.location),
        ),
        &element_type,
        array_get_vir_exprx,
    )
}

fn ast_loop_to_vir_expr(
    loop_body: &Expression,
    location: Option<Location>,
    mode: Mode,
    globals: &BTreeMap<GlobalId, (String, Type, Expression)>,
) -> Expr {
    let exprx = ExprX::Loop {
        loop_isolation: true,
        is_for_loop: false,
        label: None,
        cond: None,
        body: ast_expr_to_vir_expr(loop_body, mode, globals),
        invs: Arc::new(Vec::new()),
        decrease: Arc::new(Vec::new()),
    };

    SpannedTyped::new(
        &build_span_no_id(format!("Loop with body {}", loop_body), location),
        &make_unit_vir_type(),
        exprx,
    )
}

fn ast_while_to_vir_expr(
    while_expression: &While,
    location: Option<Location>,
    mode: Mode,
    globals: &BTreeMap<GlobalId, (String, Type, Expression)>,
) -> Expr {
    let exprx = ExprX::Loop {
        loop_isolation: true,
        is_for_loop: false,
        label: None,
        cond: Some(ast_expr_to_vir_expr(&while_expression.condition, mode, globals)),
        body: ast_expr_to_vir_expr(&while_expression.body, mode, globals),
        invs: Arc::new(Vec::new()),
        decrease: Arc::new(Vec::new()),
    };

    SpannedTyped::new(
        &build_span_no_id(
            format!(
                "While with condition {} and body {}",
                while_expression.condition, while_expression.body
            ),
            location,
        ),
        &make_unit_vir_type(),
        exprx,
    )
}

fn binary_op_to_vir_binary_op(
    ast_binary_op: &BinaryOpKind,
    mode: Mode,
    lhs_type: &Typ,
) -> BinaryOp {
    let is_lhs_bool = matches!(lhs_type.as_ref(), TypX::Bool);
    let bool_or_bitwise = |bool_op, bitwise_op| {
        if is_lhs_bool { bool_op } else { BinaryOp::Bitwise(bitwise_op, mode) }
    };

    match ast_binary_op {
        BinaryOpKind::Add => BinaryOp::Arith(ArithOp::Add, mode),
        BinaryOpKind::Subtract => BinaryOp::Arith(ArithOp::Sub, mode),
        BinaryOpKind::Multiply => BinaryOp::Arith(ArithOp::Mul, mode),
        BinaryOpKind::Divide => BinaryOp::Arith(ArithOp::EuclideanDiv, mode),
        BinaryOpKind::Modulo => BinaryOp::Arith(ArithOp::EuclideanMod, mode),

        BinaryOpKind::Equal => BinaryOp::Eq(mode),
        BinaryOpKind::NotEqual => BinaryOp::Ne,
        BinaryOpKind::Less => BinaryOp::Inequality(InequalityOp::Lt),
        BinaryOpKind::LessEqual => BinaryOp::Inequality(InequalityOp::Le),
        BinaryOpKind::Greater => BinaryOp::Inequality(InequalityOp::Gt),
        BinaryOpKind::GreaterEqual => BinaryOp::Inequality(InequalityOp::Ge),

        BinaryOpKind::And => bool_or_bitwise(BinaryOp::And, BitwiseOp::BitAnd),
        BinaryOpKind::Or => bool_or_bitwise(BinaryOp::Or, BitwiseOp::BitOr),
        BinaryOpKind::Xor => bool_or_bitwise(BinaryOp::Xor, BitwiseOp::BitXor),

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

pub fn function_name_to_vir_fun(func_name: String) -> Fun {
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

pub fn get_lvalue_ident(lvalue: &LValue) -> &Ident {
    match lvalue {
        LValue::Ident(ident) => ident,
        LValue::Index { array, .. } => get_lvalue_ident(&array),
        LValue::MemberAccess { object, .. } => get_lvalue_ident(&object),
        LValue::Dereference { reference, .. } => get_lvalue_ident(&reference),
    }
}

fn ast_lvalue_ident_to_vir_expr(ident: &Ident) -> Expr {
    let ident_id: u32 =
        ast_definition_to_id(&ident.definition).expect("Definition doesn't have an id");
    let var_ident = ast_ident_to_vir_var_ident(ident, ident_id);

    let exprx = ExprX::VarLoc(var_ident);

    SpannedTyped::new(
        &build_span(ident_id, format!("Lhs Var {}", ident.name), ident.location),
        &ast_type_to_vir_type(&ident.typ),
        exprx,
    )
}

pub fn ast_ident_to_vir_var_ident(ident: &Ident, ident_id: u32) -> VarIdent {
    // Check if we encounter the special variable with name "%return" which we produce
    // during the Monomorphization of the AST. This is the "result" variable which
    // you can refer from `ensures` attributes. We define it as AirLocal because
    // we want to differentiate it from normal variables.
    let var_disambiguate = if ident.name == FUNC_RETURN_VAR_NAME {
        VarIdentDisambiguate::AirLocal
    } else {
        VarIdentDisambiguate::RustcId(
            ident_id.try_into().expect("Failed to convert ast id to usize"),
        )
    };

    VarIdent(Arc::new(ident.name.clone()), var_disambiguate)
}

/// Panics if LValue is of type Index
fn ast_lvalue_to_vir_expr(lvalue: &LValue, location: Option<Location>, mode: Mode) -> Expr {
    match lvalue {
        LValue::Ident(ident) => ast_lvalue_ident_to_vir_expr(ident),
        LValue::Index { .. } => unreachable!(), // Array assignment has to be handled in a special way
        LValue::MemberAccess { object, field_index } => {
            let object_type = get_lvalue_type(object);
            assert!(
                matches!(object_type, Type::Tuple(_)),
                "We can only access members of type tuple!"
            );
            match object_type {
                Type::Tuple(inner_types) => {
                    let object_as_vir_expr = ast_lvalue_to_vir_expr(&object, location, mode);

                    let exprx = ExprX::UnaryOpr(
                        UnaryOpr::Field(FieldOpr {
                            datatype: Dt::Tuple(inner_types.len()),
                            variant: Arc::new(format!("tuple%{}", inner_types.len())),
                            field: Arc::new(field_index.to_string()),
                            get_variant: false,
                            check: VariantCheck::None,
                        }),
                        object_as_vir_expr,
                    );
                    SpannedTyped::new(
                        &build_span_no_id(
                            format!("Member access at index {}", field_index),
                            location,
                        ),
                        &ast_type_to_vir_type(&inner_types[*field_index]),
                        exprx,
                    )
                }
                _ => unreachable!(), // Unreachable because of the `assert!()` above
            }
        }
        LValue::Dereference { reference, .. } => {
            // There is no equivalent expression for `Dereference` in VIR.
            // Verus doesn't support lhs dereference for assignment operations.
            // We can still try to convert it as a regular lhs var and ignore the dereference
            ast_lvalue_to_vir_expr(&reference, location, mode)
        }
    }
}

fn get_lvalue_type(lvalue: &LValue) -> &Type {
    match lvalue {
        LValue::Ident(ident) => &ident.typ,
        LValue::Index { element_type, .. } => element_type,
        LValue::MemberAccess { object, field_index } => match get_lvalue_type(&object) {
            Type::Tuple(inner_types) => &inner_types[*field_index],
            object_type => object_type, //TODO(totel): This branch is perhaps unreachable?
        },
        LValue::Dereference { element_type, .. } => element_type,
    }
}

pub fn ast_definition_to_id(definition: &Definition) -> Option<u32> {
    match definition {
        Definition::Local(local_id) => Some(local_id.0),
        Definition::Global(global_id) => Some(global_id.0),
        Definition::Function(func_id) => Some(func_id.0),
        Definition::Builtin(_) | Definition::LowLevel(_) | Definition::Oracle(_) => None,
    }
}

/// For the Noir Field type we have to wrap all arithmetic instructions
/// with a Euclidean modulo `p` operation where `p` is the modulus of
/// the Noir Field.
pub fn wrap_with_field_modulo(dividend: Expr, mode: Mode) -> Expr {
    let expr_span = dividend.span.clone();
    let expr_type = dividend.typ.clone();

    let field_modulus: BigInt =
        BigInt::from_biguint(num_bigint::Sign::Plus, FieldElement::modulus());
    let divisor_expr = SpannedTyped::new(
        &expr_span,
        &Arc::new(TypX::Int(IntRange::Int)),
        ExprX::Const(Constant::Int(field_modulus)),
    );
    let modulo_exprx =
        ExprX::Binary(BinaryOp::Arith(ArithOp::EuclideanMod, mode), dividend, divisor_expr);

    SpannedTyped::new(&expr_span, &expr_type, modulo_exprx)
}
