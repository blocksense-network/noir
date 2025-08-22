use std::{convert::identity, fmt::Display, ops::AddAssign};

use crate::{
    MonomorphizationRequest, State,
    ast::{
        AnnExpr, BinaryOp, ExprF, Literal, SpannedExpr, SpannedTypedExpr, UnaryOp, Variable, cata,
        try_cata, try_contextual_cata,
    },
};
use noirc_errors::Location;
use noirc_frontend::{
    ast::IntegerBitSize,
    hir::{resolution::errors::ResolverError, type_check::TypeCheckError},
    monomorphization::{FUNC_RETURN_VAR_NAME, ast::Type as NoirType},
    shared::Signedness,
};
use num_bigint::{BigInt, BigUint};
use num_traits::{One, Zero};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OptionalType {
    /// Well-typed
    Well(NoirType),
    /// Untyped (yet) integer literal or quantified variable
    IntegerLiteral,
    /// Tuple with holes
    PartialTuple(Vec<OptionalType>),
}

impl Display for OptionalType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OptionalType::Well(t) => write!(f, "{t}"),
            OptionalType::IntegerLiteral => write!(f, "Integer literal"),
            OptionalType::PartialTuple(types) => write!(f, "Partial tuple with types: {:?}", types),
        }
    }
}

impl OptionalType {
    pub fn unwrap_or(self, or_arg: NoirType) -> NoirType {
        match self {
            OptionalType::Well(noir_typ) => noir_typ,
            OptionalType::IntegerLiteral => or_arg,
            OptionalType::PartialTuple(_optional_types) => {
                unreachable!("Partial types must have been resolved")
            }
        }
    }
}

pub type SpannedPartiallyTypedExpr = AnnExpr<(Location, OptionalType)>;

#[derive(Debug, Clone)]
pub enum TypeInferenceError {
    MonomorphizationRequest(MonomorphizationRequest),
    // NOTE: We are converting IntegerLiteralDoesNotFit to TypeCheckError later
    // We cannot do it during construction because we need a function
    // located in a module which depends on us.
    IntegerLiteralDoesNotFit {
        literal: BigInt,
        literal_type: NoirType,
        fit_into: Option<NoirType>,
        location: Location,
        message: String,
    },
    NoirTypeError(TypeCheckError),
}

#[derive(Debug, PartialEq, Eq)]
pub enum FitsIn {
    Yes,
    No {
        /// `None` means there is no possible `IntegerBitSize` that could contain us
        need: Option<NoirType>,
    },
}
pub fn bi_can_fit_in(bi: &BigInt, hole_size: &IntegerBitSize, hole_sign: &Signedness) -> FitsIn {
    let is_negative = match bi.sign() {
        num_bigint::Sign::Minus => true,
        num_bigint::Sign::Plus | num_bigint::Sign::NoSign => false,
    };

    // NOTE: if we have a negative literal, but try to fit into an unsigned integer
    let is_wrong_sign = is_negative && !hole_sign.is_signed();

    let mut bits = bi.bits();

    // NOTE: Add one bit for the sign.
    if hole_sign.is_signed() {
        bits += 1;
    }

    if is_negative {
        fn is_power_of_two(n: &BigUint) -> bool {
            // 10000000
            // 01111111
            !n.is_zero() && ((n & (n - BigUint::one())) == BigUint::zero())
        }

        // NOTE: ...unless we have a negative number whose magnitude is a power of two.
        //       This is the because it fits perfectly into the two's complement minimum value.
        //       (e.g., -128 fits in `i8`, the same as -127, but not 128).
        if is_power_of_two(bi.magnitude()) {
            bits -= 1;
        }
    }

    // NOTE: find the smallest `IntegerBitSize` that could contain us, `None` if no such one exists
    let smallest_fit_size =
        IntegerBitSize::allowed_sizes().into_iter().find(|size| bits <= size.bit_size() as u64);

    // NOTE: using the `PartialOrd` instance for `IntegerBitSize`,
    //       which definitionally orders the bit sizes in an increasing order
    let size_fits = smallest_fit_size.map(|sfs| sfs <= *hole_size).unwrap_or(false);

    if !size_fits || is_wrong_sign {
        return FitsIn::No {
            need: smallest_fit_size.map(|sfs| {
                NoirType::Integer(if is_wrong_sign { Signedness::Signed } else { *hole_sign }, sfs)
            }),
        };
    }

    return FitsIn::Yes;
}

impl SpannedPartiallyTypedExpr {
    /// Unifies a partially-typed expression with a target concrete type
    /// - If the expression is an `IntegerLiteral`, it's checked and given the target type
    /// - If the expression is already `Well`-typed, it checks if the types match
    /// - If the expression is a `PartialTuple`, it recursively unifies its elements
    pub fn unify_with_type(&mut self, target_type: NoirType) -> Result<(), TypeInferenceError> {
        match &self.ann.1 {
            OptionalType::IntegerLiteral => match self.expr.as_mut() {
                ExprF::Literal { value: Literal::Int(bi) } => {
                    if let NoirType::Integer(ref sign, ref size) = target_type
                        && let FitsIn::No { need } = bi_can_fit_in(&bi, size, sign)
                    {
                        return Err(TypeInferenceError::IntegerLiteralDoesNotFit {
                            literal: bi.clone(),
                            literal_type: target_type.clone(),
                            fit_into: need.clone(),
                            location: self.ann.0,
                            message: format!(
                                "Integer literal {} cannot fit in {}, needs at least {:?} or larger",
                                bi, target_type, need
                            ),
                        });
                    }

                    self.ann.1 = OptionalType::Well(target_type);
                }
                // NOTE: quantified variable
                ExprF::Variable(_var) => self.ann.1 = OptionalType::Well(target_type),
                // NOTE: not shifting
                ExprF::BinaryOp { op: _, expr_left, expr_right } => {
                    expr_left.unify_with_type(target_type.clone())?;
                    expr_right.unify_with_type(target_type.clone())?;
                    self.ann.1 = OptionalType::Well(target_type);
                }
                ExprF::UnaryOp { op, expr } => match op {
                    UnaryOp::Dereference => {
                        return Err(TypeInferenceError::NoirTypeError(
                            TypeCheckError::TypeMismatch {
                                expected_typ: self.ann.1.to_string(),
                                expr_typ: "Reference".to_string(),
                                expr_location: self.ann.0,
                            },
                        ));
                    }
                    UnaryOp::Not => {
                        expr.unify_with_type(target_type.clone())?;
                        self.ann.1 = OptionalType::Well(target_type);
                    }
                },
                ExprF::TupleAccess { expr: inner_tuple_expr, index } => {
                    let ExprF::Tuple { exprs } = inner_tuple_expr.expr.as_mut() else {
                        unreachable!()
                    };

                    let new_element_types = exprs
                        .into_iter()
                        .enumerate()
                        .map(|(i, elem_expr)| {
                            if i as u32 == *index {
                                elem_expr.unify_with_type(target_type.clone())?;
                            } else if matches!(elem_expr.ann.1, OptionalType::IntegerLiteral) {
                                // NOTE: Default other integer literals to Field, as they are unconstrained.
                                elem_expr.unify_with_type(NoirType::Field)?;
                            }

                            Ok(elem_expr.ann.1.clone())
                        })
                        .collect::<Result<Vec<_>, _>>()?;

                    let updated_inner_tuple_type = new_element_types
                        .iter()
                        .map(|ot| match ot {
                            OptionalType::Well(t) => Some(t.clone()),
                            _ => None,
                        })
                        .collect::<Option<Vec<_>>>()
                        .map(NoirType::Tuple)
                        .map(OptionalType::Well)
                        .unwrap_or(OptionalType::PartialTuple(new_element_types));

                    inner_tuple_expr.ann.1 = updated_inner_tuple_type;
                    self.ann.1 = OptionalType::Well(target_type);
                }
                ExprF::Parenthesised { expr: inner_expr } => {
                    inner_expr.unify_with_type(target_type.clone())?;
                    self.ann.1 = OptionalType::Well(target_type);
                }

                ExprF::Literal { value: Literal::Bool(..) }
                | ExprF::Quantified { .. }
                | ExprF::FnCall { .. }
                | ExprF::Index { .. }
                | ExprF::Cast { .. }
                | ExprF::Tuple { .. }
                | ExprF::Array { .. } => unreachable!(
                    "ICE: Unexpected expression {:?} found with IntegerLiteral type",
                    self.expr
                ),
            },

            OptionalType::Well(well_type) => {
                if *well_type != target_type {
                    return Err(TypeInferenceError::NoirTypeError(TypeCheckError::TypeMismatch {
                        expected_typ: target_type.to_string(),
                        expr_typ: well_type.to_string(),
                        expr_location: self.ann.0,
                    }));
                }

                self.ann.1 = OptionalType::Well(target_type);
            }

            OptionalType::PartialTuple(_) => {
                let NoirType::Tuple(ref target_expr_types) = target_type else {
                    return Err(TypeInferenceError::NoirTypeError(TypeCheckError::TypeMismatch {
                        expected_typ: target_type.to_string(),
                        expr_typ: "tuple".to_string(),
                        expr_location: self.ann.0,
                    }));
                };

                let ExprF::Tuple { exprs } = self.expr.as_mut() else { unreachable!() };

                if exprs.len() != target_expr_types.len() {
                    return Err(TypeInferenceError::NoirTypeError(TypeCheckError::TupleMismatch {
                        tuple_types: vec![],
                        actual_count: exprs.len(),
                        location: self.ann.0,
                    }));
                }

                std::iter::zip(exprs, target_expr_types)
                    .try_for_each(|(expr, t)| expr.unify_with_type(t.clone()))?;

                self.ann.1 = OptionalType::Well(target_type);
            }
        }

        Ok(())
    }
}

/// Converts an `OptionalType` into a concrete `NoirType`,
/// using the default for any remaining untyped integer literals,
/// unless in a partial tuple
pub fn concretize_type(
    opt_type: OptionalType,
    default_integer_literal_type: &NoirType,
) -> Option<NoirType> {
    match opt_type {
        OptionalType::Well(t) => Some(t),
        OptionalType::IntegerLiteral => Some(default_integer_literal_type.clone()),
        OptionalType::PartialTuple(elements) => elements
            .into_iter()
            .map(|e| match e {
                OptionalType::IntegerLiteral => None,
                _ => concretize_type(e, default_integer_literal_type),
            })
            .collect::<Option<_>>()
            .map(NoirType::Tuple),
    }
}

pub fn type_infer(
    state: &State,
    expr: SpannedExpr,
) -> Result<SpannedTypedExpr, TypeInferenceError> {
    // NOTE: predicate, always bool,
    //       assume subterms are `u32` (like `Noir` does)
    let default_literal_type = NoirType::Integer(Signedness::Unsigned, IntegerBitSize::ThirtyTwo);

    let is_numeric = |t: &NoirType| matches!(t, NoirType::Integer(_, _) | NoirType::Field);

    let sote: SpannedPartiallyTypedExpr = try_contextual_cata(
        expr,
        vec![],
        &|mut quantifier_bound_variables, e| {
            if let ExprF::Quantified { variables, .. } = e.expr.as_ref() {
                quantifier_bound_variables.extend(variables.iter().map(
                    |Variable { path, name, .. }| {
                        // NOTE: quantified variables should have no path
                        debug_assert_eq!(path.len(), 0);

                        // NOTE:: reserve an `id` for this variable
                        let id = state.min_local_id.borrow().clone();
                        state.min_local_id.borrow_mut().add_assign(1);

                        (id, name.clone())
                    },
                ));
            }
            quantifier_bound_variables
        },
        &|location, quantifier_bound_variables, mut exprf| {
            let exprf_type = match &mut exprf {
                ExprF::Literal { value } => match value {
                    Literal::Bool(_) => OptionalType::Well(NoirType::Bool),
                    Literal::Int(_) => {
                        // NOTE: `OptionalType::IntegerLiteral` signifies that this has to be inferred up the chain,
                        //        will gain a concrete type when it gets matched in an (arithmetic or predicate) operation
                        //        with a variable with a real (integer) type
                        OptionalType::IntegerLiteral
                    }
                },
                ExprF::Variable(Variable { path: _, name, id }) => {
                    // NOTE: Ignoring paths since they're already stripped from the provided state

                    // NOTE: parsing should not yield `id`s
                    debug_assert_eq!(*id, None);
                    let (variable_ident, variable_id, variable_type): (
                        &str,
                        Option<u32>,
                        OptionalType,
                    ) = quantifier_bound_variables
                        .iter()
                        .find_map(|(id, bound_variable)| {
                            (bound_variable == name).then(|| {
                                (name.as_str(), Some(id.clone()), OptionalType::IntegerLiteral)
                            })
                        })
                        .or_else(|| {
                            state.function.parameters.iter().find_map(|(id, _, par_name, t, _)| {
                                (par_name == name).then(|| {
                                    (name.as_str(), Some(id.0), OptionalType::Well(t.clone()))
                                })
                            })
                        })
                        .or_else(|| {
                            state.global_constants.iter().find_map(
                                |(global_id, (global_name, t, _))| {
                                    (global_name == name).then(|| {
                                        (
                                            name.as_str(),
                                            Some(global_id.0),
                                            OptionalType::Well(t.clone()),
                                        )
                                    })
                                },
                            )
                        })
                        .or_else(|| {
                            (name == "result").then(|| {
                                (
                                    FUNC_RETURN_VAR_NAME,
                                    None,
                                    OptionalType::Well(state.function.return_type.clone()),
                                )
                            })
                        })
                        .ok_or(TypeInferenceError::NoirTypeError(TypeCheckError::ResolverError(
                            ResolverError::VariableNotDeclared { name: name.to_string(), location },
                        )))?;

                    *name = variable_ident.to_string();
                    *id = variable_id;

                    variable_type
                }
                ExprF::FnCall { name, args } => {
                    let func_object = state
                        .functions
                        .iter()
                        .find_map(|(_, func)| (func.name == *name).then_some(func))
                        .ok_or(TypeInferenceError::MonomorphizationRequest(
                            MonomorphizationRequest {
                                function_identifier: name.clone(),
                                param_types: args
                                    .iter()
                                    .map(|arg: &SpannedPartiallyTypedExpr| arg.ann.1.clone())
                                    .collect(),
                            },
                        ))?;

                    // Unify the function call arguments with their expected type
                    for (arg, (_id, _mut, _name, typ, _visibility)) in
                        args.iter_mut().zip(&func_object.parameters)
                    {
                        arg.unify_with_type(typ.clone())?;
                    }

                    OptionalType::Well(func_object.return_type.clone())
                }
                ExprF::Quantified { variables, .. } => {
                    variables
                        .iter()
                        .map(|Variable { path, .. }| {
                            if !path.is_empty() {
                                Err(TypeInferenceError::NoirTypeError(
                                    // TODO(totel): better error?
                                    TypeCheckError::ParameterCountMismatch {
                                        expected: 0,
                                        found: path.len(),
                                        location,
                                    },
                                ))
                            } else {
                                Ok(())
                            }
                        })
                        .collect::<Result<Vec<_>, _>>()?;

                    // NOTE: add predertimed `id`s to the quantified variables
                    variables.iter_mut().for_each(|variable| {
                        variable.id = Some(
                            *quantifier_bound_variables
                                .iter()
                                .find_map(|(id, bound_variable)| {
                                    (*bound_variable == variable.name).then(|| id)
                                })
                                .expect("Should have been populated while traversing down the AST"),
                        );
                    });

                    OptionalType::Well(NoirType::Bool)
                }
                ExprF::Parenthesised { expr } => expr.ann.1.clone(),
                ExprF::UnaryOp { op, expr } => {
                    let t = match op {
                        UnaryOp::Dereference => {
                            match &expr.ann.1 {
                                OptionalType::Well(NoirType::Reference(t, _)) => {
                                    OptionalType::Well(*t.clone())
                                }
                                _ => {
                                    // TODO(totel): better error?
                                    return Err(TypeInferenceError::NoirTypeError(
                                        TypeCheckError::UnconstrainedReferenceToConstrained {
                                            location,
                                        },
                                    ));
                                }
                            }
                        }
                        UnaryOp::Not => expr.ann.1.clone(),
                    };

                    t
                }
                ExprF::BinaryOp { op, expr_left, expr_right } => {
                    fn is_tupleish(t: &OptionalType) -> bool {
                        matches!(
                            t,
                            OptionalType::Well(NoirType::Tuple(_)) | OptionalType::PartialTuple(_)
                        )
                    }

                    if matches!(op, BinaryOp::Eq | BinaryOp::Neq)
                        && is_tupleish(&expr_left.ann.1)
                        && is_tupleish(&expr_right.ann.1)
                    {
                        enum TupleElement<'a> {
                            Literal(&'a mut SpannedPartiallyTypedExpr, OptionalType),
                            Variable(NoirType),
                        }
                        fn tuple_elements_from_expr<'a>(
                            expr: &'a mut ExprF<SpannedPartiallyTypedExpr>,
                            ann: OptionalType,
                            location: Location,
                        ) -> Result<Vec<TupleElement<'a>>, TypeInferenceError>
                        {
                            match (expr, ann) {
                                (
                                    ExprF::Tuple { exprs },
                                    OptionalType::Well(NoirType::Tuple(types)),
                                ) => Ok(std::iter::zip(exprs, types)
                                    .map(|(e, t)| TupleElement::Literal(e, OptionalType::Well(t)))
                                    .collect()),

                                (ExprF::Tuple { exprs }, OptionalType::PartialTuple(types)) => {
                                    Ok(std::iter::zip(exprs, types)
                                        .map(|(e, t)| TupleElement::Literal(e, t))
                                        .collect())
                                }

                                (
                                    ExprF::Variable(_),
                                    OptionalType::Well(NoirType::Tuple(types)),
                                ) => Ok(types.into_iter().map(TupleElement::Variable).collect()),

                                (ExprF::Variable(_), OptionalType::PartialTuple(_)) => {
                                    // Unreachable because if we have a variable of type tuple we
                                    // would always know its type.
                                    unreachable!()
                                }

                                _ => Err(TypeInferenceError::NoirTypeError(
                                    TypeCheckError::TupleMismatch {
                                        location,
                                        tuple_types: vec![],
                                        actual_count: 0,
                                    },
                                )),
                            }
                        }

                        let left_elements: Vec<TupleElement> = tuple_elements_from_expr(
                            expr_left.expr.as_mut(),
                            expr_left.ann.1.clone(),
                            location,
                        )?;
                        let right_elements: Vec<TupleElement> = tuple_elements_from_expr(
                            expr_right.expr.as_mut(),
                            expr_right.ann.1.clone(),
                            location,
                        )?;

                        if left_elements.len() != right_elements.len() {
                            // TODO(totel): better error?
                            return Err(TypeInferenceError::NoirTypeError(
                                TypeCheckError::TupleMismatch {
                                    location,
                                    tuple_types: vec![],
                                    actual_count: left_elements.len(),
                                },
                            ));
                        }

                        // NOTE: Unify all pair of elements
                        let new_exprs_types: Vec<_> = std::iter::zip(left_elements, right_elements)
                            .map(|(left, right)| {
                                match (left, right) {
                                    (
                                        TupleElement::Literal(left_expr, left_type),
                                        TupleElement::Literal(right_expr, right_type),
                                    ) => match (left_type, right_type) {
                                        (OptionalType::Well(t1), OptionalType::Well(t2)) => {
                                            if t1 != t2 {
                                                return Err(TypeInferenceError::NoirTypeError(
                                                    TypeCheckError::TypeMismatch {
                                                        expected_typ: t1.to_string(),
                                                        expr_typ: t2.to_string(),
                                                        expr_location: right_expr.ann.0,
                                                    },
                                                ));
                                            }
                                            Ok(t1.clone())
                                        }
                                        (OptionalType::Well(t), OptionalType::IntegerLiteral) => {
                                            right_expr.unify_with_type(t.clone())?;
                                            Ok(t.clone())
                                        }
                                        (OptionalType::IntegerLiteral, OptionalType::Well(t)) => {
                                            left_expr.unify_with_type(t.clone())?;
                                            Ok(t.clone())
                                        }
                                        (
                                            OptionalType::IntegerLiteral,
                                            OptionalType::IntegerLiteral,
                                        ) => {
                                            left_expr
                                                .unify_with_type(default_literal_type.clone())?;
                                            right_expr
                                                .unify_with_type(default_literal_type.clone())?;
                                            Ok(default_literal_type.clone())
                                        }
                                        // NOTE: All other combinations (e.g., trying to unify a bool with a tuple) are a type mismatch.
                                        //       The calling function is responsible for handling recursive cases like tuples.
                                        (t1, t2) => Err(TypeInferenceError::NoirTypeError(
                                            TypeCheckError::TypeMismatch {
                                                expected_typ: t1.to_string(),
                                                expr_typ: t2.to_string(),
                                                expr_location: right_expr.ann.0,
                                            },
                                        )),
                                    },
                                    (
                                        TupleElement::Literal(left_expr, _left_type),
                                        TupleElement::Variable(right_type),
                                    ) => {
                                        left_expr.unify_with_type(right_type.clone())?;
                                        Ok(right_type)
                                    }
                                    (
                                        TupleElement::Variable(left_type),
                                        TupleElement::Literal(right_expr, _right_type),
                                    ) => {
                                        right_expr.unify_with_type(left_type.clone())?;
                                        Ok(left_type)
                                    }
                                    (
                                        TupleElement::Variable(left_type),
                                        TupleElement::Variable(right_type),
                                    ) => {
                                        if left_type != right_type {
                                            return Err(TypeInferenceError::NoirTypeError(
                                                TypeCheckError::TypeMismatch {
                                                    expected_typ: right_type.to_string(),
                                                    expr_typ: left_type.to_string(),
                                                    expr_location: location,
                                                },
                                            ));
                                        }

                                        Ok(left_type)
                                    }
                                }
                            })
                            .collect::<Result<Vec<_>, _>>()?;

                        expr_left.ann.1 =
                            OptionalType::Well(NoirType::Tuple(new_exprs_types.clone()));
                        expr_right.ann.1 = OptionalType::Well(NoirType::Tuple(new_exprs_types));

                        OptionalType::Well(NoirType::Bool)
                    } else if op.is_shift() {
                        let shift_amount_type =
                            NoirType::Integer(Signedness::Unsigned, IntegerBitSize::Eight);

                        match &expr_right.ann.1 {
                            OptionalType::Well(t) if *t == shift_amount_type => {} // OK
                            OptionalType::IntegerLiteral => {
                                expr_right.unify_with_type(shift_amount_type)?;
                            }
                            OptionalType::Well(_) | OptionalType::PartialTuple(_) => {
                                return Err(TypeInferenceError::NoirTypeError(
                                    TypeCheckError::InvalidShiftSize { location },
                                ));
                            }
                        }

                        if let OptionalType::IntegerLiteral = &expr_left.ann.1 {
                            expr_left.unify_with_type(default_literal_type.clone())?;
                        }

                        expr_left.ann.1.clone()
                    } else {
                        match (&expr_left.ann.1, &expr_right.ann.1) {
                            (OptionalType::Well(t1), OptionalType::Well(t2)) => {
                                if t1 != t2 {
                                    return Err(TypeInferenceError::NoirTypeError(
                                        TypeCheckError::TypeMismatch {
                                            expected_typ: t1.to_string(),
                                            expr_typ: t2.to_string(),
                                            expr_location: expr_right.ann.0,
                                        },
                                    ));
                                }

                                OptionalType::Well(if op.is_comparison() {
                                    NoirType::Bool
                                } else {
                                    t1.clone()
                                })
                            }

                            (OptionalType::Well(t1), OptionalType::IntegerLiteral) => {
                                if (op.is_arithmetic() || op.is_bitwise()) && !is_numeric(t1) {
                                    return Err(TypeInferenceError::NoirTypeError(
                                        TypeCheckError::TypeMismatch {
                                            expected_typ: "a numeric type".to_string(),
                                            expr_typ: t1.to_string(),
                                            expr_location: expr_left.ann.0,
                                        },
                                    ));
                                }

                                expr_right.unify_with_type(t1.clone())?;

                                OptionalType::Well(if op.is_comparison() {
                                    NoirType::Bool
                                } else {
                                    t1.clone()
                                })
                            }

                            (OptionalType::IntegerLiteral, OptionalType::Well(t2)) => {
                                if (op.is_arithmetic() || op.is_bitwise()) && !is_numeric(t2) {
                                    return Err(TypeInferenceError::NoirTypeError(
                                        TypeCheckError::TypeMismatch {
                                            expected_typ: "a numeric type".to_string(),
                                            expr_typ: t2.to_string(),
                                            expr_location: expr_right.ann.0,
                                        },
                                    ));
                                }

                                expr_left.unify_with_type(t2.clone())?;

                                OptionalType::Well(if op.is_comparison() {
                                    NoirType::Bool
                                } else {
                                    t2.clone()
                                })
                            }

                            (OptionalType::IntegerLiteral, OptionalType::IntegerLiteral) => {
                                if op.is_arithmetic() || op.is_bitwise() {
                                    OptionalType::IntegerLiteral
                                } else {
                                    expr_left.unify_with_type(default_literal_type.clone())?;
                                    expr_right.unify_with_type(default_literal_type.clone())?;

                                    OptionalType::Well(NoirType::Bool)
                                }
                            }

                            // NOTE: Any other combination involving tuples is an error in this arm
                            (t1, t2) => {
                                return Err(TypeInferenceError::NoirTypeError(
                                    TypeCheckError::TypeMismatch {
                                        expected_typ: t1.to_string(),
                                        expr_typ: t2.to_string(),
                                        expr_location: location,
                                    },
                                ));
                            }
                        }
                    }
                }
                ExprF::Index { expr, index } => {
                    let OptionalType::Well(NoirType::Array(_, type_inner)) = &expr.ann.1 else {
                        return Err(TypeInferenceError::NoirTypeError(
                            TypeCheckError::TypeMismatch {
                                expr_typ: expr.ann.1.to_string(),
                                expected_typ: String::from("Array type"),
                                expr_location: location,
                            },
                        ));
                    };
                    match &index.ann.1 {
                        // Fine index
                        OptionalType::Well(NoirType::Integer(Signedness::Unsigned, _)) => {}
                        // Integer literal, try type inferring to `u32`
                        OptionalType::IntegerLiteral => {
                            index.unify_with_type(default_literal_type.clone())?;
                        }
                        // Not fine index
                        t => {
                            return Err(TypeInferenceError::NoirTypeError(
                                TypeCheckError::TypeMismatch {
                                    expr_typ: t.to_string(),
                                    expected_typ: String::from("Unsigned integer type"),
                                    expr_location: location,
                                },
                            ));
                        }
                    }

                    OptionalType::Well(*type_inner.clone())
                }
                ExprF::TupleAccess { expr, index } => {
                    let t = match &expr.ann.1 {
                        OptionalType::Well(NoirType::Tuple(types)) => {
                            types.get(*index as usize).cloned().map(OptionalType::Well).ok_or(
                                TypeInferenceError::NoirTypeError(
                                    TypeCheckError::TupleIndexOutOfBounds {
                                        index: *index as usize,
                                        // NOTE: We are converting to Type::Tuple of empty vec because
                                        // the inner types don't really matter for the error reporting
                                        lhs_type: noirc_frontend::Type::Tuple(vec![]),
                                        length: types.len(),
                                        location,
                                    },
                                ),
                            )?
                        }
                        OptionalType::PartialTuple(types) => {
                            types.get(*index as usize).cloned().ok_or(
                                TypeInferenceError::NoirTypeError(
                                    TypeCheckError::TupleIndexOutOfBounds {
                                        index: *index as usize,
                                        // NOTE: We are converting to Type::Tuple of empty vec because
                                        // the inner types don't really matter for the error reporting
                                        lhs_type: noirc_frontend::Type::Tuple(vec![]),
                                        length: types.len(),
                                        location,
                                    },
                                ),
                            )?
                        }
                        _ => {
                            // TODO(totel): better error?
                            return Err(TypeInferenceError::NoirTypeError(
                                TypeCheckError::ResolverError(ResolverError::SelfReferentialType {
                                    location,
                                }),
                            ));
                        }
                    };

                    t
                }
                ExprF::Cast { expr, target } => {
                    // Non-booleans cannot cast to bool
                    if matches!(target, NoirType::Bool)
                        && !matches!(expr.ann.1, OptionalType::Well(NoirType::Bool))
                    {
                        return Err(TypeInferenceError::NoirTypeError(
                            TypeCheckError::CannotCastNumericToBool {
                                // NOTE: We are using a mock type, because we can't convert to wanted type
                                typ: noirc_frontend::Type::FieldElement,
                                location,
                            },
                        ));
                    }

                    // Non-numerics cannot cast to numeric types
                    if is_numeric(target)
                        && let OptionalType::Well(ref t) = expr.ann.1
                        && !is_numeric(t)
                        && !matches!(t, NoirType::Bool)
                    {
                        return Err(TypeInferenceError::NoirTypeError(
                            TypeCheckError::TypeMismatch {
                                expected_typ: String::from("Numeric or a boolean type"),
                                expr_typ: t.to_string(),
                                expr_location: location,
                            },
                        ));
                    }

                    // Try to type infer integer literals as the target type
                    if matches!(expr.ann.1, OptionalType::IntegerLiteral) {
                        expr.unify_with_type(target.clone())?;
                    }

                    OptionalType::Well(target.clone())
                }
                ExprF::Tuple { exprs } => {
                    // NOTE: if all sub-expressions are well-typed, produce a well-typed `Tuple`
                    //       otherwise, produce a `OptionalType::PartialTuple`
                    let t = exprs
                        .iter()
                        .map(|e| e.ann.1.clone())
                        .map(|ot| match ot {
                            OptionalType::Well(t) => Some(t),
                            _ => None,
                        })
                        .collect::<Option<Vec<_>>>()
                        .map(NoirType::Tuple)
                        .map(OptionalType::Well)
                        .unwrap_or(OptionalType::PartialTuple(
                            exprs.iter().map(|ae| ae.ann.1.clone()).collect(),
                        ));

                    t
                }
                ExprF::Array { exprs } => {
                    if exprs.is_empty() {
                        // TODO(totel): better error?
                        return Err(TypeInferenceError::NoirTypeError(
                            TypeCheckError::InvalidTypeForEntryPoint { location },
                        ));
                    }

                    let unified_opt_type = exprs.iter().try_fold(
                        OptionalType::IntegerLiteral,
                        |acc, current_expr| {
                            // This closure acts as our "join" operation.
                            fn join_types(
                                t1: OptionalType,
                                t2: OptionalType,
                                location: Location,
                            ) -> Result<OptionalType, TypeInferenceError>
                            {
                                match (t1, t2) {
                                    // An integer literal can join with any other type.
                                    (t, OptionalType::IntegerLiteral) => Ok(t),
                                    (OptionalType::IntegerLiteral, t) => Ok(t),

                                    // If both types are well-known, they must be identical.
                                    (OptionalType::Well(w1), OptionalType::Well(w2)) => {
                                        if w1 == w2 {
                                            Ok(OptionalType::Well(w1))
                                        } else {
                                            Err(TypeInferenceError::NoirTypeError(
                                                // TODO: calculate indices
                                                TypeCheckError::NonHomogeneousArray {
                                                    first_type: w1.to_string(),
                                                    first_location: location,
                                                    first_index: 0,
                                                    second_type: w2.to_string(),
                                                    second_location: location,
                                                    second_index: 0,
                                                },
                                            ))
                                        }
                                    }

                                    // Recursively join partial tuples.
                                    (
                                        OptionalType::PartialTuple(v1),
                                        OptionalType::PartialTuple(v2),
                                    ) => {
                                        if v1.len() != v2.len() {
                                            /* TODO: Mismatch error */
                                            Ok(OptionalType::PartialTuple(vec![]))
                                        } else {
                                            let joined = v1
                                                .into_iter()
                                                .zip(v2.into_iter())
                                                .map(|(e1, e2)| join_types(e1, e2, location))
                                                .collect::<Result<_, _>>()?;
                                            Ok(OptionalType::PartialTuple(joined))
                                        }
                                    }

                                    // Coerce Well(Tuple) to PartialTuple for joining.
                                    (
                                        OptionalType::Well(NoirType::Tuple(v1)),
                                        t2 @ OptionalType::PartialTuple(_),
                                    ) => join_types(
                                        OptionalType::PartialTuple(
                                            v1.into_iter().map(OptionalType::Well).collect(),
                                        ),
                                        t2,
                                        location,
                                    ),
                                    (
                                        t1 @ OptionalType::PartialTuple(_),
                                        OptionalType::Well(NoirType::Tuple(v2)),
                                    ) => join_types(
                                        t1,
                                        OptionalType::PartialTuple(
                                            v2.into_iter().map(OptionalType::Well).collect(),
                                        ),
                                        location,
                                    ),

                                    // NOTE: All other combinations are a non-homogeneous array error
                                    (other1, other2) => Err(TypeInferenceError::NoirTypeError(
                                        TypeCheckError::NonHomogeneousArray {
                                            first_type: other1.to_string(),
                                            first_location: location,
                                            first_index: 0,
                                            second_type: other2.to_string(),
                                            second_location: location,
                                            second_index: 0,
                                        },
                                    )),
                                }
                            }
                            join_types(acc, current_expr.ann.1.clone(), location)
                        },
                    )?;

                    let concrete_element_type =
                        concretize_type(unified_opt_type, &default_literal_type).ok_or(
                            TypeInferenceError::NoirTypeError(
                                TypeCheckError::TypeAnnotationNeededOnArrayLiteral {
                                    is_array: true,
                                    location,
                                },
                            ),
                        )?;

                    exprs
                        .into_iter()
                        .try_for_each(|expr| expr.unify_with_type(concrete_element_type.clone()))?;

                    OptionalType::Well(NoirType::Array(
                        exprs.len() as u32,
                        Box::new(concrete_element_type),
                    ))
                }
            };

            Ok(SpannedPartiallyTypedExpr { ann: (location, exprf_type), expr: Box::new(exprf) })
        },
    )?;

    let fully_typed_expr: SpannedTypedExpr =
        try_cata(sote, &|(location, otype), exprf| match otype {
            OptionalType::Well(t) => {
                Ok(SpannedTypedExpr { ann: (location, t), expr: Box::new(exprf) })
            }
            _ => Err(format!("Expr {:?} still has no type ({:?})", exprf, otype)),
        })
        .expect("Typing should have either succeeded or have resulted in an expected error");

    // NOTE: only the `FUNC_RETURN_VAR_NAME` variable should have no id
    assert!(cata(fully_typed_expr.clone(), &|ann, exprf| match exprf {
        ExprF::Variable(Variable { path: _, name, id }) => {
            let res = if name == FUNC_RETURN_VAR_NAME { id.is_none() } else { id.is_some() };
            if !res {
                dbg!(ann, name, id);
            }
            res
        }

        ExprF::FnCall { args, .. } => args.into_iter().all(identity),
        ExprF::Quantified { expr, .. } => expr,
        ExprF::Parenthesised { expr } => expr,
        ExprF::UnaryOp { expr, .. } => expr,
        ExprF::BinaryOp { expr_left, expr_right, .. } => expr_left && expr_right,
        ExprF::Index { expr, index } => expr && index,
        ExprF::TupleAccess { expr, .. } => expr,
        ExprF::Cast { expr, .. } => expr,
        ExprF::Tuple { exprs } => exprs.into_iter().all(identity),
        ExprF::Array { exprs } => exprs.into_iter().all(identity),

        // Non-recursive variants don't carry information
        ExprF::Literal { .. } => true,
    }));

    Ok(fully_typed_expr)
}

#[cfg(test)]
mod tests {
    use noirc_errors::{Location, Span};
    use noirc_frontend::ast::IntegerBitSize;
    use noirc_frontend::shared::Signedness;
    use num_traits::Num;
    use std::convert::identity;

    use super::*;

    use crate::{
        Attribute,
        ast::{Literal, Variable, cata, strip_ann},
        parse::{parse_attribute, tests::*},
    };

    #[test]
    fn test_integer_types() {
        {
            let bi127 = BigInt::from_str_radix("127", 10).unwrap();
            let bi128 = BigInt::from_str_radix("128", 10).unwrap();
            let bi129 = BigInt::from_str_radix("129", 10).unwrap();
            let bin127 = BigInt::from_str_radix("-127", 10).unwrap();
            let bin128 = BigInt::from_str_radix("-128", 10).unwrap();
            let bin129 = BigInt::from_str_radix("-129", 10).unwrap();

            let hole_size = IntegerBitSize::Eight;
            let hole_sign = Signedness::Signed;

            let bi127_fit = bi_can_fit_in(&bi127, &hole_size, &hole_sign);
            let bi128_fit = bi_can_fit_in(&bi128, &hole_size, &hole_sign);
            let bi129_fit = bi_can_fit_in(&bi129, &hole_size, &hole_sign);
            let bin127_fit = bi_can_fit_in(&bin127, &hole_size, &hole_sign);
            let bin128_fit = bi_can_fit_in(&bin128, &hole_size, &hole_sign);
            let bin129_fit = bi_can_fit_in(&bin129, &hole_size, &hole_sign);

            assert_eq!(bi127_fit, FitsIn::Yes);
            assert_eq!(
                bi128_fit,
                FitsIn::No { need: Some(NoirType::Integer(hole_sign, IntegerBitSize::Sixteen)) }
            );
            assert_eq!(
                bi129_fit,
                FitsIn::No { need: Some(NoirType::Integer(hole_sign, IntegerBitSize::Sixteen)) }
            );

            assert_eq!(bin127_fit, FitsIn::Yes);
            assert_eq!(bin128_fit, FitsIn::Yes);
            assert_eq!(
                bin129_fit,
                FitsIn::No { need: Some(NoirType::Integer(hole_sign, IntegerBitSize::Sixteen)) }
            );
        }

        {
            let bi255 = BigInt::from_str_radix("255", 10).unwrap();
            let bi256 = BigInt::from_str_radix("256", 10).unwrap();
            let bi257 = BigInt::from_str_radix("257", 10).unwrap();
            let bin1 = BigInt::from_str_radix("-1", 10).unwrap();

            let hole_size = IntegerBitSize::Eight;
            let hole_sign = Signedness::Unsigned;

            let bi255_fit = bi_can_fit_in(&bi255, &hole_size, &hole_sign);
            let bi256_fit = bi_can_fit_in(&bi256, &hole_size, &hole_sign);
            let bi257_fit = bi_can_fit_in(&bi257, &hole_size, &hole_sign);
            let bin1_fit = bi_can_fit_in(&bin1, &hole_size, &hole_sign);

            assert_eq!(bi255_fit, FitsIn::Yes);
            assert_eq!(
                bi256_fit,
                FitsIn::No { need: Some(NoirType::Integer(hole_sign, IntegerBitSize::Sixteen)) }
            );
            assert_eq!(
                bi257_fit,
                FitsIn::No { need: Some(NoirType::Integer(hole_sign, IntegerBitSize::Sixteen)) }
            );

            assert_eq!(
                bin1_fit,
                FitsIn::No {
                    need: Some(NoirType::Integer(Signedness::Signed, IntegerBitSize::One))
                }
            );
        }
    }

    #[test]
    fn test_whole_attribute() {
        let attribute = "ensures(result >= a + (16 / 2 % (7 * 4)))";
        let state = empty_state();
        let attribute = parse_attribute(
            attribute,
            Location { span: Span::inclusive(0, attribute.len() as u32), file: Default::default() },
            state.function,
            state.global_constants,
            state.functions,
        )
        .unwrap();
        let Attribute::Ensures(spanned_expr) = attribute else { panic!() };
        let spanned_typed_expr = type_infer(&state, spanned_expr).unwrap();
        assert!(
            cata(spanned_typed_expr, &|(_, expr_type), expr| {
                match expr {
                    ExprF::Literal { value: Literal::Int(_) } => {
                        expr_type
                            == NoirType::Integer(Signedness::Signed, IntegerBitSize::ThirtyTwo)
                    }
                    ExprF::FnCall { args, .. } => args.into_iter().all(identity),
                    ExprF::Quantified { expr, .. } => expr,
                    ExprF::Parenthesised { expr } => expr,
                    ExprF::UnaryOp { expr, .. } => expr,
                    ExprF::BinaryOp { expr_left, expr_right, .. } => expr_left && expr_right,
                    ExprF::Index { expr, index } => expr && index,
                    ExprF::TupleAccess { expr, .. } => expr,
                    ExprF::Cast { expr, .. } => expr,
                    ExprF::Tuple { exprs } => exprs.into_iter().all(identity),
                    ExprF::Array { exprs } => exprs.into_iter().all(identity),

                    // Non-recursive variants don't carry information
                    ExprF::Literal { value: Literal::Bool(_) } | ExprF::Variable(_) => true,
                }
            }),
            "All integer literals have the correct inferred type"
        );
    }

    #[test]
    fn test_quantifiers() {
        let attribute = "ensures(exists(| i ,  j | result >= a + (16 / i % (7 * 4))))";
        let state = empty_state();
        let attribute = parse_attribute(
            attribute,
            Location { span: Span::inclusive(0, attribute.len() as u32), file: Default::default() },
            state.function,
            state.global_constants,
            state.functions,
        )
        .unwrap();
        let Attribute::Ensures(spanned_expr) = attribute else { panic!() };
        let spanned_typed_expr = type_infer(&state, spanned_expr).unwrap();
        assert!(
            cata(spanned_typed_expr.clone(), &|(_, expr_type), expr| {
                match expr {
                    ExprF::Literal { value: Literal::Int(_) } => {
                        expr_type
                            == NoirType::Integer(Signedness::Signed, IntegerBitSize::ThirtyTwo)
                    }

                    ExprF::FnCall { args, .. } => args.into_iter().all(identity),
                    ExprF::Quantified { expr, .. } => expr,
                    ExprF::Parenthesised { expr } => expr,
                    ExprF::UnaryOp { expr, .. } => expr,
                    ExprF::BinaryOp { expr_left, expr_right, .. } => expr_left && expr_right,
                    ExprF::Index { expr, index } => expr && index,
                    ExprF::TupleAccess { expr, .. } => expr,
                    ExprF::Cast { expr, .. } => expr,
                    ExprF::Tuple { exprs } => exprs.into_iter().all(identity),
                    ExprF::Array { exprs } => exprs.into_iter().all(identity),

                    // Non-recursive variants don't carry information
                    ExprF::Literal { value: Literal::Bool(_) } | ExprF::Variable(_) => true,
                }
            }),
            "All integer literals have the correct inferred type"
        );
        assert!(
            cata(spanned_typed_expr, &|(_, expr_type), expr| {
                match expr {
                    ExprF::Variable(Variable { name, .. }) => {
                        if name == "i" {
                            expr_type
                                == NoirType::Integer(Signedness::Signed, IntegerBitSize::ThirtyTwo)
                        } else {
                            true
                        }
                    }

                    ExprF::FnCall { args, .. } => args.into_iter().all(identity),
                    ExprF::Quantified { expr, .. } => expr,
                    ExprF::Parenthesised { expr } => expr,
                    ExprF::UnaryOp { expr, .. } => expr,
                    ExprF::BinaryOp { expr_left, expr_right, .. } => expr_left && expr_right,
                    ExprF::Index { expr, index } => expr && index,
                    ExprF::TupleAccess { expr, .. } => expr,
                    ExprF::Cast { expr, .. } => expr,
                    ExprF::Tuple { exprs } => exprs.into_iter().all(identity),
                    ExprF::Array { exprs } => exprs.into_iter().all(identity),

                    // Non-recursive variants don't carry information
                    ExprF::Literal { .. } => true,
                }
            }),
            "All bound variables have the correct inferred type"
        );
    }

    #[test]
    fn test_index() {
        let attribute = "ensures(xs[1 + 1] > 5)";
        let state = empty_state();
        let attribute = parse_attribute(
            attribute,
            Location { span: Span::inclusive(0, attribute.len() as u32), file: Default::default() },
            state.function,
            state.global_constants,
            state.functions,
        )
        .unwrap();
        let Attribute::Ensures(spanned_expr) = attribute else { panic!() };
        let spanned_typed_expr = type_infer(&state, spanned_expr).unwrap();
        dbg!(&spanned_typed_expr);
        assert_eq!(spanned_typed_expr.ann.1, NoirType::Bool);
    }

    #[test]
    fn test_dereference_index() {
        let attribute = "ensures((*rxs)[1 + 1] > 5)";
        let state = empty_state();
        let attribute = parse_attribute(
            attribute,
            Location { span: Span::inclusive(0, attribute.len() as u32), file: Default::default() },
            state.function,
            state.global_constants,
            state.functions,
        )
        .unwrap();
        let Attribute::Ensures(spanned_expr) = attribute else { panic!() };
        let spanned_typed_expr = type_infer(&state, spanned_expr).unwrap();
        dbg!(&spanned_typed_expr);
        assert_eq!(spanned_typed_expr.ann.1, NoirType::Bool);
    }

    #[test]
    fn test_operators_mixed_types() {
        let attribute = "ensures(1 + true)";
        let state = empty_state();
        let attribute = parse_attribute(
            attribute,
            Location { span: Span::inclusive(0, attribute.len() as u32), file: Default::default() },
            state.function,
            state.global_constants,
            state.functions,
        )
        .unwrap();
        let Attribute::Ensures(spanned_expr) = attribute else { panic!() };
        let type_inference_error = type_infer(&state, spanned_expr).unwrap_err();
        let TypeInferenceError::NoirTypeError(TypeCheckError::TypeMismatch {
            expected_typ,
            expr_typ,
            expr_location,
        }) = type_inference_error
        else {
            panic!()
        };
        dbg!(&expr_typ, &expected_typ, &expr_location);

        // NOTE: untyped integer literal (same for quantifier variables) force the other argument
        //       to also be numeric
        assert_eq!(expr_typ, "bool");
        assert_eq!(expected_typ, "a numeric type");
    }

    #[test]
    fn test_bitshift() {
        let attribute = "ensures(1 << 256)";
        let state = empty_state();
        let attribute = parse_attribute(
            attribute,
            Location { span: Span::inclusive(0, attribute.len() as u32), file: Default::default() },
            state.function,
            state.global_constants,
            state.functions,
        )
        .unwrap();
        let Attribute::Ensures(spanned_expr) = attribute else { panic!() };
        let type_inference_error = type_infer(&state, spanned_expr).unwrap_err();
        let TypeInferenceError::IntegerLiteralDoesNotFit {
            literal: _,
            literal_type,
            fit_into,
            location: _,
            message,
        } = type_inference_error
        else {
            panic!()
        };
        dbg!(&literal_type, &fit_into, &message);
        assert_eq!(literal_type, NoirType::Integer(Signedness::Unsigned, IntegerBitSize::Eight));
        // NOTE: minimal size that fits `256`
        assert_eq!(
            fit_into,
            Some(NoirType::Integer(Signedness::Unsigned, IntegerBitSize::Sixteen))
        );
        assert_eq!(message, "Integer literal 256 cannot fit in u8, needs at least Some(Integer(Unsigned, Sixteen)) or larger".to_string());
    }

    #[test]
    fn test_tuple_access() {
        let attribute = "ensures(user.0 ==> true)";
        let state = empty_state();
        let attribute = parse_attribute(
            attribute,
            Location { span: Span::inclusive(0, attribute.len() as u32), file: Default::default() },
            state.function,
            state.global_constants,
            state.functions,
        )
        .unwrap();
        let Attribute::Ensures(spanned_expr) = attribute else { panic!() };
        let spanned_typed_expr = type_infer(&state, spanned_expr).unwrap();
        dbg!(&spanned_typed_expr);
        assert_eq!(spanned_typed_expr.ann.1, NoirType::Bool);
    }

    #[test]
    fn test_tuple_access_combos() {
        let attribute = "ensures(exists(|i| (0 <= i) & (i < 20) & xs[i] > 100))";
        let state = empty_state();
        let attribute = parse_attribute(
            attribute,
            Location { span: Span::inclusive(0, attribute.len() as u32), file: Default::default() },
            state.function,
            state.global_constants,
            state.functions,
        )
        .unwrap();
        let Attribute::Ensures(spanned_expr) = attribute else { panic!() };
        let spanned_typed_expr = type_infer(&state, spanned_expr).unwrap();
        dbg!(&spanned_typed_expr);
        assert_eq!(spanned_typed_expr.ann.1, NoirType::Bool);
    }

    #[test]
    fn test_unary_op_literals() {
        let attribute = "ensures(result as Field != !15)";
        let state = empty_state();
        let attribute = parse_attribute(
            attribute,
            Location { span: Span::inclusive(0, attribute.len() as u32), file: Default::default() },
            state.function,
            state.global_constants,
            state.functions,
        )
        .unwrap();
        let Attribute::Ensures(spanned_expr) = attribute else { panic!() };
        let spanned_typed_expr = type_infer(&state, spanned_expr).unwrap();
        dbg!(&spanned_typed_expr);
        assert_eq!(spanned_typed_expr.ann.1, NoirType::Bool);
    }

    #[test]
    fn test_cast_simple() {
        let annotation = "ensures(15 == 1 as u8)";
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

        let Attribute::Ensures(spanned_expr) = attribute else { panic!() };
        let spanned_typed_expr = type_infer(&state, spanned_expr).unwrap();
        dbg!(&strip_ann(spanned_typed_expr));
        // assert_eq!(spanned_typed_expr.ann.1, NoirType::Bool);
    }

    #[test]
    fn test_cast() {
        let annotation = "ensures((15 as i16 - 3 > 2) & ((result as Field - 6) as u64 == 1 + a as u64 >> 4 as u8))";
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

        let Attribute::Ensures(spanned_expr) = attribute else { panic!() };
        let spanned_typed_expr = type_infer(&state, spanned_expr).unwrap();
        dbg!(&strip_ann(spanned_typed_expr));
        // assert_eq!(spanned_typed_expr.ann.1, NoirType::Bool);
    }

    #[test]
    fn test_double_cast() {
        let annotation = "ensures(a == (a as i16) as i32)";
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

        let Attribute::Ensures(spanned_expr) = attribute else { panic!() };
        let spanned_typed_expr = type_infer(&state, spanned_expr).unwrap();
        dbg!(&strip_ann(spanned_typed_expr));
        // assert_eq!(spanned_typed_expr.ann.1, NoirType::Bool);
    }

    #[test]
    fn test_tuple_complex() {
        let annotation = "ensures((1000, 5, 1000).1 == 15 as u8)";
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

        let Attribute::Ensures(spanned_expr) = attribute else { panic!() };
        let spanned_typed_expr = type_infer(&state, spanned_expr).unwrap();
        dbg!(&strip_ann(spanned_typed_expr));
    }

    #[test]
    fn test_tuple() {
        let annotation = "ensures(((), kek, true).2)";
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

        let Attribute::Ensures(spanned_expr) = attribute else { panic!() };
        let spanned_typed_expr = type_infer(&state, spanned_expr).unwrap();
        dbg!(&strip_ann(spanned_typed_expr));
    }

    #[test]
    fn test_partial_tuple_equality() {
        let annotation = "ensures((1 as u8, 15) == (1, 15 as i16))";
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

        let Attribute::Ensures(spanned_expr) = attribute else { panic!() };
        let spanned_typed_expr = type_infer(&state, spanned_expr).unwrap();
        dbg!(&strip_ann(spanned_typed_expr.clone()));

        let ExprF::BinaryOp { op: BinaryOp::Eq, expr_left, expr_right } = *spanned_typed_expr.expr
        else {
            panic!()
        };

        let ExprF::Tuple { exprs: _ } = *expr_left.expr else { panic!() };
        let ExprF::Tuple { exprs: _ } = *expr_right.expr else { panic!() };

        // Assert that both tuple types are fully unified
        assert_eq!(
            expr_left.ann.1,
            NoirType::Tuple(vec![
                NoirType::Integer(Signedness::Unsigned, IntegerBitSize::Eight),
                NoirType::Integer(Signedness::Signed, IntegerBitSize::Sixteen)
            ])
        );
        assert_eq!(expr_left.ann.1, expr_right.ann.1);
    }

    #[test]
    fn test_tuple_equality() {
        let annotation = "ensures((1, 15) == pair)";
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

        let Attribute::Ensures(spanned_expr) = attribute else { panic!() };
        let spanned_typed_expr = type_infer(&state, spanned_expr).unwrap();
        dbg!(&strip_ann(spanned_typed_expr.clone()));

        let ExprF::BinaryOp { op: BinaryOp::Eq, expr_left, expr_right } = *spanned_typed_expr.expr
        else {
            panic!()
        };

        let ExprF::Tuple { exprs: _ } = *expr_left.expr else { panic!() };
        let ExprF::Variable(_) = *expr_right.expr else { panic!() };

        // Assert that both tuple types are fully unified
        assert_eq!(
            expr_left.ann.1,
            NoirType::Tuple(vec![
                NoirType::Integer(Signedness::Unsigned, IntegerBitSize::Sixteen),
                NoirType::Field,
            ])
        );
        assert_eq!(expr_left.ann.1, expr_right.ann.1);
    }

    #[test]
    fn test_array_equality() {
        let annotation = "ensures([(1, 2), (1 as u8, 7), (8, 9 as i16)] == [(0, 0), (1 as u8, 2 as i16), (0, 0)])";
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

        let Attribute::Ensures(spanned_expr) = attribute else { panic!() };
        let spanned_typed_expr = type_infer(&state, spanned_expr).unwrap();
        dbg!(&strip_ann(spanned_typed_expr.clone()));

        let ExprF::BinaryOp { op: BinaryOp::Eq, expr_left, expr_right } = *spanned_typed_expr.expr
        else {
            panic!()
        };

        let ExprF::Array { exprs: _ } = *expr_left.expr else { panic!() };
        let ExprF::Array { exprs: _ } = *expr_right.expr else { panic!() };

        // Assert that both array types are fully unified
        assert_eq!(
            expr_left.ann.1,
            NoirType::Array(
                3,
                Box::new(NoirType::Tuple(vec![
                    NoirType::Integer(Signedness::Unsigned, IntegerBitSize::Eight),
                    NoirType::Integer(Signedness::Signed, IntegerBitSize::Sixteen)
                ]))
            )
        );
        assert_eq!(expr_left.ann.1, expr_right.ann.1);
    }

    #[test]
    fn test_array() {
        let annotation = "ensures([15 as i32, a, 129 as i32][2])";
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

        let Attribute::Ensures(spanned_expr) = attribute else { panic!() };
        let spanned_typed_expr = type_infer(&state, spanned_expr).unwrap();
        dbg!(&strip_ann(spanned_typed_expr));
    }

    #[test]
    fn test_monomorphization_request() {
        let attribute = "ensures(f(result))";
        let state = empty_state();
        let attribute = parse_attribute(
            attribute,
            Location { span: Span::inclusive(0, attribute.len() as u32), file: Default::default() },
            state.function,
            state.global_constants,
            state.functions,
        )
        .unwrap();
        let Attribute::Ensures(spanned_expr) = attribute else { panic!() };
        let type_inference_error = type_infer(&state, spanned_expr).unwrap_err();
        let TypeInferenceError::MonomorphizationRequest(MonomorphizationRequest {
            function_identifier,
            param_types,
        }) = type_inference_error
        else {
            panic!()
        };
        assert_eq!(function_identifier, "f");
        assert_eq!(
            param_types,
            vec![OptionalType::Well(NoirType::Integer(
                Signedness::Signed,
                IntegerBitSize::ThirtyTwo
            ))]
        );
    }
}
