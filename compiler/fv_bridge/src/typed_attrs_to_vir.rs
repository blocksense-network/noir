use std::sync::Arc;

use formal_verification::{
    State,
    ast::{BinaryOp, ExprF, Quantifier, SpannedTypedExpr, Variable, cata},
};
use noirc_evaluator::vir::vir_gen::{
    Attribute, build_span, build_span_no_id,
    expr_to_vir::{
        expr::{function_name_to_vir_fun, numeric_const_to_vir_exprx, wrap_with_field_modulo},
        std_functions::handle_fv_std_call_in_annotations,
        types::{
            ast_const_to_vir_type_const, ast_type_to_vir_type, get_bit_not_bitwidth, is_type_field,
        },
    },
};
use noirc_frontend::monomorphization::FUNC_RETURN_VAR_NAME;
use vir::{
    ast::{
        AirQuant, BinaryOp as VirBinaryOp, BinderX, BitwiseOp, Dt, FieldOpr, FunX, ImplPath,
        InequalityOp, IntRange, PathX, Primitive, Quant, Typ, UnaryOpr, VarBinder, VarBinderX,
        VarIdent, VarIdentDisambiguate, VariantCheck,
    },
    ast_util::int_range_from_type,
};
use vir::{
    ast::{
        ArithOp, AutospecUsage, CallTarget, CallTargetKind, Constant, Expr, ExprX, Mode,
        SpannedTyped, TypX,
    },
    ast_util::{bitwidth_from_type, is_integer_type_signed},
};

use crate::TypedAttribute;

use acvm::{AcirField, FieldElement};
use noirc_frontend::signed_field::SignedField;
use num_bigint::{BigInt, ToBigInt}; // Replace with the actual path to SignedField

pub fn signed_field_from_bigint_wrapping(value: BigInt) -> SignedField {
    let modulus = FieldElement::modulus().to_bigint().unwrap();

    // Wrap value to the positive modulus range
    let wrapped = ((&value % &modulus) + &modulus) % &modulus;

    // Convert the reduced BigInt into FieldElement
    let wrapped_biguint = wrapped.to_biguint().unwrap();
    let bytes = wrapped_biguint.to_bytes_be();
    let field = FieldElement::from_be_bytes_reduce(&bytes);
    // Fields are always positive
    let is_negative = false;
    SignedField::new(field, is_negative)
}

pub(crate) fn convert_typed_attribute_to_vir_attribute(
    typed_attribute: TypedAttribute,
    state: &State,
) -> Attribute {
    match typed_attribute {
        TypedAttribute::Ghost => Attribute::Ghost,
        TypedAttribute::Requires(ann_expr) => {
            Attribute::Requires(ann_expr_to_vir_expr(ann_expr, state))
        }
        TypedAttribute::Ensures(ann_expr) => {
            Attribute::Ensures(ann_expr_to_vir_expr(ann_expr, state))
        }
    }
}

pub(crate) fn ann_expr_to_vir_expr(ann_expr: SpannedTypedExpr, state: &State) -> Expr {
    let mode = Mode::Spec;
    cata(ann_expr, &|(loc, typ), expr| -> Arc<SpannedTyped<ExprX>> {
        // Helper to construct a standard SpannedTyped expression.
        // It captures `loc` from the outer scope.
        let make_expr = |exprx: ExprX, vir_type: Typ, span_msg: String| -> Expr {
            SpannedTyped::new(&build_span_no_id(span_msg, Some(loc)), &vir_type, exprx)
        };
        match expr {
            ExprF::Literal { value } => {
                let (exprx, span_msg) = match value {
                    formal_verification::ast::Literal::Bool(val) => {
                        (ExprX::Const(Constant::Bool(val)), format!("Const boolean {}", val))
                    }
                    formal_verification::ast::Literal::Int(big_int) => (
                        numeric_const_to_vir_exprx(
                            &signed_field_from_bigint_wrapping(big_int.clone()),
                            &typ,
                        ),
                        format!("Int literal {}", big_int),
                    ),
                };
                make_expr(exprx, ast_type_to_vir_type(&typ), span_msg)
            }
            ExprF::Variable(Variable { name, id, path: _ }) => {
                let is_global = state.global_constants.iter().any(|(_, (gn, _, _))| *gn == name);
                let span_msg = format!("Var {}", name);

                // Handle the simpler global case first and return early.
                if is_global {
                    let exprx = ExprX::ConstVar(
                        Arc::new(FunX {
                            path: Arc::new(PathX {
                                krate: None,
                                segments: Arc::new(vec![Arc::new(name)]),
                            }),
                        }),
                        AutospecUsage::Final,
                    );

                    // Globals don't have a specific noir DefId, so use build_span_no_id.
                    return SpannedTyped::new(
                        &build_span_no_id(span_msg, Some(loc)),
                        &ast_type_to_vir_type(&typ),
                        exprx,
                    );
                }

                // --- Logic below is now only for non-global (local) variables ---

                let disambiguate = if name == FUNC_RETURN_VAR_NAME {
                    VarIdentDisambiguate::AirLocal
                } else {
                    let rustc_id = id
                        .expect("Non-return variable must have an ID")
                        .try_into()
                        .expect("Failed to convert ast id to usize");
                    VarIdentDisambiguate::RustcId(rustc_id)
                };

                let exprx = ExprX::Var(VarIdent(Arc::new(name), disambiguate));

                // Idiomatically create the span based on whether an ID exists.
                let span = if let Some(def_id) = id {
                    build_span(def_id, span_msg, Some(loc))
                } else {
                    build_span_no_id(span_msg, Some(loc))
                };

                SpannedTyped::new(&span, &ast_type_to_vir_type(&typ), exprx)
            }
            ExprF::FnCall { name, args } => {
                // TODO(totel): Special handling for `old` from the Noir `fv_std`

                if let Some(expr) =
                    handle_fv_std_call_in_annotations(&name, &args, loc, &typ)
                {
                    return expr;
                }

                let exprx = ExprX::Call(
                    CallTarget::Fun(
                        CallTargetKind::Static,
                        function_name_to_vir_fun(name.clone()),
                        Arc::new(vec![]),
                        Arc::new(vec![]),
                        AutospecUsage::Final,
                    ),
                    Arc::new(args),
                );
                make_expr(exprx, ast_type_to_vir_type(&typ), format!("Function call {}", name))
            }
            ExprF::Quantified { quantifier, variables, expr } => {
                let quantifier_type = match quantifier {
                    Quantifier::Forall => Quant { quant: AirQuant::Forall },
                    Quantifier::Exists => Quant { quant: AirQuant::Exists },
                };
                let make_binder = |ident: &Variable| -> VarBinder<Typ> {
                    let rustc_id = ident
                        .id
                        .expect("Quantifier index should have a local id")
                        .try_into()
                        .expect("Failed to convert u32 id to usize");
                    Arc::new(VarBinderX {
                        name: VarIdent(
                            Arc::new(ident.name.clone()),
                            VarIdentDisambiguate::RustcId(rustc_id),
                        ),
                        a: Arc::new(TypX::Int(IntRange::Int)),
                    })
                };
                let quantifier_vir_indices = Arc::new(variables.iter().map(make_binder).collect());

                assert!(
                    matches!(expr.typ.as_ref(), TypX::Bool),
                    "All quantifier bodies must be of type bool"
                );

                let exprx = ExprX::Quant(quantifier_type, quantifier_vir_indices, expr);
                make_expr(exprx, Arc::new(TypX::Bool), format!("Quantifier {}", quantifier))
            }
            ExprF::Parenthesised { expr } => expr,
            ExprF::UnaryOp { op, expr } => {
                let (exprx, span_msg) = match op {
                    formal_verification::ast::UnaryOp::Not => {
                        if matches!(typ, noirc_frontend::monomorphization::ast::Type::Bool) {
                            (
                                ExprX::Unary(vir::ast::UnaryOp::Not, expr),
                                "Logical Not op".to_string(),
                            )
                        } else {
                            (
                                ExprX::Unary(
                                    vir::ast::UnaryOp::BitNot(get_bit_not_bitwidth(&typ)),
                                    expr,
                                ),
                                "BitNot expr".to_string(),
                            )
                        }
                    }
                    formal_verification::ast::UnaryOp::Dereference => {
                        // If we have Dereference(Expr), Verus treats them as if it is only Expr.
                        // Also the expr type gets trimmed from the reference "decoration".
                        // Therefore we will do the same and only change the type of the expression.
                        let deref_expr = SpannedTyped::new(
                            &build_span_no_id(
                                format!("Dereference expression with type {}", typ),
                                Some(loc),
                            ),
                            &ast_type_to_vir_type(&typ),
                            expr.x.clone(), // Can not move out of Arc
                        );

                        return deref_expr;
                    }
                };
                make_expr(exprx, ast_type_to_vir_type(&typ), span_msg)
            }
            ExprF::BinaryOp { op, expr_left, expr_right } => {
                let lhs_type = expr_left.typ.clone();
                let is_lhs_bool = matches!(lhs_type.as_ref(), TypX::Bool);
                let bool_or_bitwise = |bool_op, bitwise_op| {
                    if is_lhs_bool { bool_op } else { VirBinaryOp::Bitwise(bitwise_op, mode) }
                };

                let vir_binary_op = match op {
                    BinaryOp::Mul => VirBinaryOp::Arith(ArithOp::Mul, mode),
                    BinaryOp::Div => VirBinaryOp::Arith(ArithOp::EuclideanDiv, mode),
                    BinaryOp::Mod => VirBinaryOp::Arith(ArithOp::EuclideanMod, mode),
                    BinaryOp::Add => VirBinaryOp::Arith(ArithOp::Add, mode),
                    BinaryOp::Sub => VirBinaryOp::Arith(ArithOp::Sub, mode),
                    BinaryOp::ShiftLeft => VirBinaryOp::Bitwise(
                        BitwiseOp::Shl(
                            bitwidth_from_type(&lhs_type)
                                .expect("Bitwise operation with non int type"),
                            is_integer_type_signed(&lhs_type),
                        ),
                        mode,
                    ),
                    BinaryOp::ShiftRight => VirBinaryOp::Bitwise(
                        BitwiseOp::Shr(
                            bitwidth_from_type(&lhs_type)
                                .expect("Bitwise operation with non int type"),
                        ),
                        mode,
                    ),
                    BinaryOp::Eq => VirBinaryOp::Eq(mode),
                    BinaryOp::Neq => VirBinaryOp::Ne,
                    BinaryOp::Lt => VirBinaryOp::Inequality(InequalityOp::Lt),
                    BinaryOp::Le => VirBinaryOp::Inequality(InequalityOp::Le),
                    BinaryOp::Gt => VirBinaryOp::Inequality(InequalityOp::Gt),
                    BinaryOp::Ge => VirBinaryOp::Inequality(InequalityOp::Ge),
                    BinaryOp::Implies => VirBinaryOp::Implies,
                    BinaryOp::And => bool_or_bitwise(VirBinaryOp::And, BitwiseOp::BitAnd),
                    BinaryOp::Or => bool_or_bitwise(VirBinaryOp::Or, BitwiseOp::BitOr),
                    BinaryOp::Xor => bool_or_bitwise(VirBinaryOp::Xor, BitwiseOp::BitXor),
                };

                let binary_op_type = ast_type_to_vir_type(&typ);
                let exprx = ExprX::Binary(vir_binary_op, expr_left, expr_right);

                let vir_binary_expr =
                    make_expr(exprx, binary_op_type, "Binary op expression".to_string());

                if is_type_field(&typ) && op.is_arithmetic() {
                    // Wrap expression with `% Field::modulus()` if the expression type is Field
                    wrap_with_field_modulo(vir_binary_expr, mode)
                } else {
                    vir_binary_expr
                }
            }
            ExprF::TupleAccess { expr, index } => {
                let TypX::Datatype(Dt::Tuple(tuple_len), _, _) = expr.typ.as_ref() else {
                    unreachable!("Unexpected type for tuple access: {:#?}", expr);
                };

                let exprx = ExprX::UnaryOpr(
                    UnaryOpr::Field(FieldOpr {
                        datatype: Dt::Tuple(*tuple_len),
                        variant: Arc::new(format!("tuple%{}", tuple_len)),
                        field: Arc::new(index.to_string()),
                        get_variant: false,
                        check: VariantCheck::None,
                    }),
                    expr,
                );

                make_expr(exprx, ast_type_to_vir_type(&typ), format!("Tuple access at {}", index))
            }
            ExprF::Index { expr, index } => {
                let element_type = ast_type_to_vir_type(&typ);

                // Ensure we're indexing into a proper array with a const length
                let array_type = match expr.typ.as_ref() {
                    TypX::Primitive(Primitive::Array, inner) => inner,
                    _ => unreachable!("Arrays must be of type Primitive"),
                };

                let array_len = match array_type[1].as_ref() {
                    TypX::ConstInt(len) => len,
                    _ => unreachable!("Arrays must have a constant length"),
                };

                let array_len_type = ast_const_to_vir_type_const(
                    array_len.try_into().expect("Failed to convert u32 to usize"),
                );

                // Build types for the vstd::array::spec_index function
                let vstd_krate = Some(Arc::new("vstd".to_string()));
                let element_and_len = Arc::new(vec![element_type.clone(), array_len_type.clone()]);
                let array_typ =
                    Arc::new(TypX::Primitive(Primitive::Array, element_and_len.clone()));
                let typs_for_call = Arc::new(vec![array_typ.clone(), element_type.clone()]);

                // Define the call target and trait impl paths
                let make_path = |segments: Vec<&str>| {
                    Arc::new(PathX {
                        krate: vstd_krate.clone(),
                        segments: Arc::new(
                            segments.into_iter().map(|s| Arc::new(s.to_string())).collect(),
                        ),
                    })
                };

                let call_target = CallTarget::Fun(
                    CallTargetKind::DynamicResolved {
                        resolved: Arc::new(FunX {
                            path: make_path(vec!["array", "impl&%2", "spec_index"]),
                        }),
                        typs: array_type.clone(),
                        impl_paths: Arc::new(vec![]),
                        is_trait_default: false,
                    },
                    Arc::new(FunX {
                        path: make_path(vec!["array", "ArrayAdditionalSpecFns", "spec_index"]),
                    }),
                    typs_for_call.clone(),
                    Arc::new(vec![
                        ImplPath::TraitImplPath(make_path(vec!["array", "impl&%0"])),
                        ImplPath::TraitImplPath(make_path(vec!["array", "impl&%2"])),
                    ]),
                    AutospecUsage::IfMarked,
                );

                let vir_exprx = ExprX::Call(call_target, Arc::new(vec![expr.clone(), index]));

                make_expr(vir_exprx, element_type, "Array indexing expression".to_string())
            }
            ExprF::Cast { expr: castee, target } => {
                let target_type = ast_type_to_vir_type(&target);
                // The following unwrap is safe because the semantic analysis of
                // the compiler should guarantee correctly typed expressions.
                let target_int_range =
                    int_range_from_type(&target_type).expect("Can not cast to a non integer type");

                let exprx = ExprX::Unary(
                    vir::ast::UnaryOp::Clip {
                        range: target_int_range,
                        truncate: false, // We are not truncating because Verus doesn't truncate casts
                    },
                    castee,
                );

                SpannedTyped::new(
                    &build_span_no_id(
                        format!("Cast expression to target type {}", target),
                        Some(loc),
                    ),
                    &target_type,
                    exprx,
                )
            }
            ExprF::Tuple { exprs } => {
                let tuple_length = exprs.len();

                let tuple_exprx = ExprX::Ctor(
                    Dt::Tuple(tuple_length),
                    Arc::new(format!("tuple%{tuple_length}")),
                    Arc::new(
                        exprs
                            .into_iter()
                            .enumerate()
                            .map(|(index, tuple_expr)| {
                                Arc::new(BinderX {
                                    name: Arc::new(index.to_string()),
                                    a: tuple_expr,
                                })
                            })
                            .collect(),
                    ),
                    None,
                );

                SpannedTyped::new(
                    &build_span_no_id(format!("Tuple constructor expression"), Some(loc)),
                    &ast_type_to_vir_type(&typ),
                    tuple_exprx,
                )
            }
            ExprF::Array { exprs } => {
                let exprx = ExprX::ArrayLiteral(Arc::new(exprs));
                SpannedTyped::new(
                    &build_span_no_id(format!("Array literal expression"), Some(loc)),
                    &ast_type_to_vir_type(&typ),
                    exprx,
                )
            }
        }
    })
}
