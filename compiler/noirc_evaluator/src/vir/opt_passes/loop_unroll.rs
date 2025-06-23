use std::collections::HashMap;

use acvm::FieldElement;
use noirc_errors::Location;
use noirc_frontend::{
    ast::{BinaryOpKind, UnaryOp},
    monomorphization::ast::{Definition, Expression, Function, Let, Literal, LocalId, Program},
    signed_field::SignedField,
};

use crate::vir::vir_gen::expr_to_vir::expression_location;

/// Unrolls all `for` loops in constrained functions.
pub fn unroll_for_loops_pass(program: &mut Program) {
    program
        .functions
        .iter_mut()
        .filter(|function| !function.unconstrained)
        .for_each(unroll_for_loops);
}

fn unroll_for_loops(function: &mut Function) {
    let mut constants: ConstScope = ConstScope::new();
    visit_expr(&mut function.body, &mut constants);
}

/// Recursively descents the AST and unrolls all encountered `for` loops.
/// This function panics if it fails to unroll a given `for` loop.
fn visit_expr(expr: &mut Expression, constants: &mut ConstScope) {
    let expression_location = expression_location(expr);
    match expr {
        Expression::Block(exprs) => {
            constants.push_scope();
            for e in exprs.iter_mut() {
                visit_expr(e, constants);
            }
            constants.pop_scope();
        }
        Expression::Let(let_expr) => {
            visit_expr(&mut *let_expr.expression, constants);
            let const_val = collect_constant_from_expression(&let_expr.expression, &constants);
            const_val.map(|val| constants.insert(let_expr.id, val));
        }
        Expression::While(while_expr) => {
            visit_expr(&mut *while_expr.condition, constants);
            visit_expr(&mut *while_expr.body, constants);
        }
        Expression::For(for_expr) => {
            // Evaluate bounds
            let start_range = collect_constant_from_expression(&for_expr.start_range, constants)
                .expect("All For loops in constrained functions must be bounded.");
            let end_range = collect_constant_from_expression(&for_expr.end_range, constants)
                .expect("All For loops in constrained functions must be bounded.");

            let start: i128 =
                start_range.try_to_signed().expect("Ranges should be convertible to i128");
            let end: i128 =
                end_range.try_to_signed().expect("Ranges should be convertible to i128");

            let mut unrolled: Vec<Expression> = Vec::new();
            constants.push_scope();

            for i in start..end {
                // Shadow loop variable with current i
                let i_as_field = SignedField::new(FieldElement::from(i), false);
                constants.insert(for_expr.index_variable, i_as_field.clone());

                let mut cloned_body = for_expr.block.clone();
                visit_expr(&mut cloned_body, constants);
                // Add `let` expression which defines the `for` index variable for
                // the unrolled block
                unrolled.push(Expression::Let(Let {
                    id: for_expr.index_variable,
                    mutable: false,
                    name: for_expr.index_name.clone(),
                    expression: Box::new(Expression::Literal(Literal::Integer(
                        i_as_field,
                        for_expr.index_type.clone(),
                        expression_location.unwrap_or(Location::dummy()),
                    ))),
                }));
                unrolled.push(*cloned_body);
            }

            constants.pop_scope();

            *expr = Expression::Block(unrolled);
        }
        Expression::If(if_expr) => {
            visit_expr(&mut *if_expr.condition, constants);
            visit_expr(&mut *if_expr.consequence, constants);
            if let Some(ref mut else_branch) = if_expr.alternative {
                visit_expr(else_branch, constants);
            }
        }
        Expression::Unary(unary) => visit_expr(&mut unary.rhs, constants),
        Expression::Binary(binary) => {
            visit_expr(&mut binary.lhs, constants);
            visit_expr(&mut binary.rhs, constants);
        }
        Expression::Index(index) => {
            visit_expr(&mut index.collection, constants);
            visit_expr(&mut index.index, constants);
        }
        Expression::Cast(cast) => visit_expr(&mut cast.lhs, constants),
        Expression::Loop(loop_body) => visit_expr(loop_body, constants),
        Expression::Match(match_expr) => {
            match_expr
                .cases
                .iter_mut()
                .for_each(|match_case| visit_expr(&mut match_case.branch, constants));
            match_expr
                .default_case
                .as_mut()
                .map(|default_case| visit_expr(default_case, constants));
        }
        Expression::Tuple(expressions) => {
            expressions
                .iter_mut()
                .for_each(|inner_expression| visit_expr(inner_expression, constants));
        }
        Expression::ExtractTupleField(expression, _) => visit_expr(expression, constants),
        Expression::Call(call) => {
            visit_expr(&mut call.func, constants);
            call.arguments.iter_mut().for_each(|argument| visit_expr(argument, constants));
        }
        Expression::Constrain(expression, ..) => visit_expr(expression, constants),
        Expression::Assign(assign) => visit_expr(&mut assign.expression, constants),
        Expression::Semi(expression) => visit_expr(expression, constants),
        Expression::Clone(expression) => visit_expr(expression, constants),
        Expression::Drop(expression) => visit_expr(expression, constants),
        Expression::Break | Expression::Continue => (),
        Expression::Quant(..) => (), // Quantifiers can't have `for` loops in them
        _ => {}
    }
}

/// For a given expression return the constant which defines it
/// if there is such a constant.
fn collect_constant_from_expression(
    expr: &Expression,
    constants: &ConstScope,
) -> Option<SignedField> {
    match expr {
        Expression::Ident(ident) => constants.get(&get_local_id(&ident.definition)?),
        Expression::Literal(literal) => match literal {
            Literal::Integer(signed_field, ..) => Some(*signed_field),
            _ => None,
        },
        Expression::Block(expressions) => expressions.last().and_then(|last_expression| {
            collect_constant_from_expression(last_expression, constants)
        }),
        Expression::Unary(unary) => collect_constant_from_expression(&unary.rhs, constants)
            .and_then(|constant| match unary.operator {
                UnaryOp::Minus => Some(-constant),
                UnaryOp::Not => None,
                UnaryOp::Reference { .. } => Some(constant),
                UnaryOp::Dereference { .. } => Some(constant),
            }),
        Expression::Binary(binary) => {
            let lhs_constant = collect_constant_from_expression(&binary.lhs, constants);
            let rhs_constant = collect_constant_from_expression(&binary.rhs, constants);
            match (lhs_constant, rhs_constant) {
                (None, None) | (None, Some(_)) | (Some(_), None) => None,
                (Some(lhs_constant), Some(rhs_constant)) => match binary.operator {
                    BinaryOpKind::Add => Some(lhs_constant + rhs_constant),
                    BinaryOpKind::Subtract => Some(lhs_constant - rhs_constant),
                    BinaryOpKind::Multiply => Some(lhs_constant * rhs_constant),
                    BinaryOpKind::Divide => Some(lhs_constant / rhs_constant),
                    BinaryOpKind::And => Some(SignedField::new(
                        FieldElement::from(
                            lhs_constant.try_to_signed::<i128>()?
                                & rhs_constant.try_to_signed::<i128>()?,
                        ),
                        lhs_constant.is_negative(),
                    )),
                    BinaryOpKind::Or => Some(SignedField::new(
                        FieldElement::from(
                            lhs_constant.try_to_signed::<i128>()?
                                | rhs_constant.try_to_signed::<i128>()?,
                        ),
                        lhs_constant.is_negative(),
                    )),
                    BinaryOpKind::Xor => Some(SignedField::new(
                        FieldElement::from(
                            lhs_constant.try_to_signed::<i128>()?
                                ^ rhs_constant.try_to_signed::<i128>()?,
                        ),
                        lhs_constant.is_negative(),
                    )),
                    BinaryOpKind::ShiftRight => Some(SignedField::new(
                        FieldElement::from(
                            lhs_constant.try_to_signed::<i128>()?
                                >> rhs_constant.try_to_signed::<i128>()?,
                        ),
                        lhs_constant.is_negative(),
                    )),
                    BinaryOpKind::ShiftLeft => Some(SignedField::new(
                        FieldElement::from(
                            lhs_constant.try_to_signed::<i128>()?
                                << rhs_constant.try_to_signed::<i128>()?,
                        ),
                        lhs_constant.is_negative(),
                    )),
                    BinaryOpKind::Modulo => Some(SignedField::new(
                        FieldElement::from(
                            lhs_constant.try_to_signed::<i128>()?
                                % rhs_constant.try_to_signed::<i128>()?,
                        ),
                        lhs_constant.is_negative(),
                    )),
                    _ => None,
                },
            }
        }
        Expression::Index(_index) => None, //TODO(totel): Constants from collections
        Expression::Cast(cast) => collect_constant_from_expression(&cast.lhs, constants),
        Expression::If(_) => None, //TODO(totel)
        Expression::Match(_) => None, //TODO(totel)
        Expression::Tuple(_expressions) => None, //TODO(totel): Constants from collections
        Expression::ExtractTupleField(_expression, _) => None, //TODO(totel): Constants from collections
        Expression::Call(_) => None, // This can be implemented only after constant propagation optimization pass
        Expression::Let(_) => unreachable!("Expressions of type `let` can't be rhs expression"),
        Expression::Assign(assign) => {
            collect_constant_from_expression(&assign.expression, constants)
        }
        Expression::Semi(expression) => collect_constant_from_expression(&expression, constants),
        Expression::Clone(expression) => collect_constant_from_expression(&expression, constants),
        Expression::Drop(expression) => collect_constant_from_expression(&expression, constants),
        _ => None,
    }
}

fn get_local_id(definition: &Definition) -> Option<LocalId> {
    match definition {
        Definition::Local(local_id) => Some(*local_id),
        _ => None,
    }
}

struct ConstScope {
    scopes: Vec<HashMap<LocalId, SignedField>>,
}

impl ConstScope {
    fn new() -> Self {
        Self { scopes: vec![HashMap::new()] }
    }

    fn push_scope(&mut self) {
        self.scopes.push(HashMap::new());
    }

    fn pop_scope(&mut self) {
        self.scopes.pop();
    }

    fn insert(&mut self, id: LocalId, value: SignedField) {
        if let Some(scope) = self.scopes.last_mut() {
            scope.insert(id, value);
        }
    }

    fn get(&self, id: &LocalId) -> Option<SignedField> {
        for scope in self.scopes.iter().rev() {
            if let Some(val) = scope.get(id) {
                return Some(*val);
            }
        }
        None
    }
}
