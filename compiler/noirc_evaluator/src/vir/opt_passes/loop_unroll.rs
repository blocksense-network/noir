use std::collections::HashMap;

use acvm::FieldElement;
use noirc_errors::Location;
use noirc_frontend::{
    ast::{BinaryOpKind, UnaryOp},
    monomorphization::ast::{
        ArrayLiteral, Definition, Expression, Function, Let, Literal, LocalId, Program,
    },
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
    visit_expr(&mut function.body, &mut constants, None);
}

/// Recursively descents the AST and unrolls all encountered `for` loops.
/// This function panics if it fails to unroll a given `for` loop.
fn visit_expr(expr: &mut Expression, constants: &mut ConstScope, let_local_id: Option<LocalId>) {
    let expression_location = expression_location(expr);
    match expr {
        Expression::Block(exprs) => {
            constants.push_scope();

            if let Some((last, rest)) = exprs.split_last_mut() {
                for e in rest {
                    visit_expr(e, constants, let_local_id);
                }

                visit_expr(last, constants, let_local_id);

                // Save the value which the block returns if possible
                let last_block_value = collect_any_constant_value(last, constants);
                constants.insert_last_block_value(last_block_value);
            }
            constants.pop_scope();
        }
        Expression::Let(let_expr) => {
            visit_expr(&mut let_expr.expression, constants, Some(let_expr.id));
            insert_collection_if_any(let_expr.id, &let_expr.expression, constants);
            let const_val = collect_constant_from_expression(&let_expr.expression, &constants);
            if let Some(val) = const_val {
                constants.insert(let_expr.id, val);
            }
        }
        Expression::While(while_expr) => {
            visit_expr(&mut while_expr.condition, constants, let_local_id);
            visit_expr(&mut while_expr.body, constants, let_local_id);
        }
        Expression::For(for_expr) => {
            // Evaluate bounds
            let start_range = collect_constant_from_expression(&for_expr.start_range, constants)
                .expect("All `for` loops in constrained functions must be bounded.");
            let end_range = collect_constant_from_expression(&for_expr.end_range, constants)
                .expect("All `for` loops in constrained functions must be bounded.");

            let start: i128 =
                start_range.try_to_signed().expect("Ranges should be convertible to i128");
            let end: i128 =
                end_range.try_to_signed().expect("Ranges should be convertible to i128");

            let mut unrolled: Vec<Expression> = Vec::new();
            constants.push_scope();

            for i in start..end {
                let i_as_field = SignedField::new(FieldElement::from(i), false);
                constants.insert(for_expr.index_variable, i_as_field.clone());

                let mut cloned_body = for_expr.block.clone();
                visit_expr(&mut cloned_body, constants, let_local_id);
                // Add `let` expression which defines the `for` index variable for
                // the unrolled block
                let let_expr_index_i = Expression::Let(Let {
                    id: for_expr.index_variable,
                    mutable: false,
                    name: for_expr.index_name.clone(),
                    expression: Box::new(Expression::Literal(Literal::Integer(
                        i_as_field,
                        for_expr.index_type.clone(),
                        expression_location.unwrap_or(Location::dummy()),
                    ))),
                });
                unrolled.push(Expression::Block(vec![let_expr_index_i, *cloned_body]));
            }

            constants.pop_scope();

            *expr = Expression::Block(unrolled);
        }
        Expression::If(if_expr) => {
            visit_expr(&mut if_expr.condition, constants, let_local_id);
            visit_expr(&mut if_expr.consequence, constants, let_local_id);
            if let Some(ref mut else_branch) = if_expr.alternative {
                visit_expr(else_branch, constants, let_local_id);
            }
        }
        Expression::Unary(unary) => visit_expr(&mut unary.rhs, constants, let_local_id),
        Expression::Binary(binary) => {
            visit_expr(&mut binary.lhs, constants, let_local_id);
            visit_expr(&mut binary.rhs, constants, let_local_id);
        }
        Expression::Index(index) => {
            visit_expr(&mut index.collection, constants, let_local_id);
            visit_expr(&mut index.index, constants, let_local_id);
        }
        Expression::Cast(cast) => visit_expr(&mut cast.lhs, constants, let_local_id),
        Expression::Loop(loop_body) => visit_expr(loop_body, constants, let_local_id),
        Expression::Match(match_expr) => {
            match_expr
                .cases
                .iter_mut()
                .for_each(|match_case| visit_expr(&mut match_case.branch, constants, let_local_id));
            match_expr
                .default_case
                .as_mut()
                .map(|default_case| visit_expr(default_case, constants, let_local_id));
        }
        Expression::Tuple(expressions) => {
            expressions
                .iter_mut()
                .for_each(|inner_expression| visit_expr(inner_expression, constants, let_local_id));
        }
        Expression::ExtractTupleField(expression, _) => {
            visit_expr(expression, constants, let_local_id)
        }
        Expression::Call(call) => {
            visit_expr(&mut call.func, constants, let_local_id);
            call.arguments
                .iter_mut()
                .for_each(|argument| visit_expr(argument, constants, let_local_id));
        }
        Expression::Constrain(expression, ..) => visit_expr(expression, constants, let_local_id),
        Expression::Assign(assign) => visit_expr(&mut assign.expression, constants, let_local_id),
        Expression::Semi(expression) => visit_expr(expression, constants, let_local_id),
        Expression::Clone(expression) => visit_expr(expression, constants, let_local_id),
        Expression::Drop(expression) => visit_expr(expression, constants, let_local_id),
        Expression::Break | Expression::Continue => (),
        Expression::Quant(..) => (), // Quantifiers can't have `for` loops in them
        _ => (),
    }
}

fn insert_collection_if_any(id: LocalId, expression: &Expression, constants: &mut ConstScope) {
    match expression {
        Expression::Literal(Literal::Array(array_literal))
        | Expression::Literal(Literal::Slice(array_literal)) => {
            constants.insert_collection(
                id,
                array_literal
                    .contents
                    .iter()
                    .map(|element| collect_constant_from_expression(element, constants))
                    .collect(),
            );
        }
        Expression::Tuple(expressions) => {
            constants.insert_collection(
                id,
                expressions
                    .iter()
                    .map(|element| collect_constant_from_expression(element, constants))
                    .collect(),
            );
        }
        Expression::Block(_) => match constants.try_to_consume_collection_last_block_value() {
            Some(constant_collection) => constants.insert_collection(id, constant_collection),
            None => (),
        },
        _ => (),
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
            _ => None, // Handled by `collect_constant_collection_from_expression`
        },
        Expression::Block(_) => match constants.get_last_block_value() {
            Some(LastBlockValue::ConstantBinding(constant_binding)) => *constant_binding,
            Some(_) | None => None,
        },
        Expression::Unary(unary) => collect_constant_from_expression(&unary.rhs, constants)
            .and_then(|constant| match unary.operator {
                UnaryOp::Minus => Some(-constant),
                UnaryOp::Not => None, //TODO(totel): Bit not implementation for SignedField required
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
                    _ => None, // Other operators return Boolean types which can not be used as a `for` range
                },
            }
        }
        Expression::Index(index_expr) => {
            let index: usize = collect_constant_from_expression(&index_expr.index, constants)?
                .try_to_unsigned()
                .expect("Array indexes should be convertible to i128");

            collect_constant_collection_from_expression(&index_expr.collection, constants)?
                .into_iter()
                .nth(index)
                .flatten()
        }
        Expression::Cast(cast) => collect_constant_from_expression(&cast.lhs, constants),
        Expression::If(if_expr) => {
            let condition = collect_constant_from_expression(&if_expr.condition, constants)?;
            if condition != SignedField::zero() {
                collect_constant_from_expression(&if_expr.consequence, constants)
            } else {
                if_expr.alternative.as_ref().and_then(|alternative_expr| {
                    collect_constant_from_expression(alternative_expr, constants)
                })
            }
        }
        Expression::Match(_) => None, //TODO(totel): Should be implemented after we start converting `match` expressions to VIR
        Expression::Tuple(_) => None, // Handled by `collect_constant_collection_from_expression()`
        Expression::ExtractTupleField(expression, index) => {
            collect_constant_collection_from_expression(expression, constants)?
                .into_iter()
                .nth(*index)
                .flatten()
        }
        Expression::Call(_) => None, //TODO(totel): This can be implemented only after constant propagation optimization pass
        Expression::Let(_) => None,
        Expression::Assign(assign) => {
            collect_constant_from_expression(&assign.expression, constants)
        }
        Expression::Semi(expression) => collect_constant_from_expression(&expression, constants),
        Expression::Clone(expression) => collect_constant_from_expression(&expression, constants),
        Expression::Drop(expression) => collect_constant_from_expression(&expression, constants),
        Expression::For(_)
        | Expression::Loop(_)
        | Expression::While(_)
        | Expression::Constrain(..)
        | Expression::Break
        | Expression::Continue
        | Expression::Quant(..) => None, // Those expressions can not return values therefore there are no constants to collect
    }
}

/// For a given expression return the collection which defines it.
/// Returns `None` if such a collection is not found.
fn collect_constant_collection_from_expression(
    expr: &Expression,
    constants: &ConstScope,
) -> Option<Vec<Option<SignedField>>> {
    match expr {
        Expression::Ident(ident) => {
            constants.get_collection(&get_local_id(&ident.definition)?).cloned()
        }
        Expression::Block(_) => match constants.get_last_block_value() {
            Some(LastBlockValue::ConstantCollection(constant_collection)) => {
                Some(constant_collection.clone()) // TODO(totel): Get rid of this clone
            }
            Some(_) | None => None,
        },
        Expression::Literal(Literal::Array(ArrayLiteral { contents, .. }))
        | Expression::Literal(Literal::Slice(ArrayLiteral { contents, .. }))
        | Expression::Tuple(contents) => Some(
            contents
                .iter()
                .map(|element| collect_constant_from_expression(element, constants))
                .collect(),
        ),
        _ => None,
    }
}

fn collect_any_constant_value(expr: &Expression, constants: &ConstScope) -> Option<LastBlockValue> {
    match (
        collect_constant_from_expression(expr, constants),
        collect_constant_collection_from_expression(expr, constants),
    ) {
        (None, None) => None,
        (None, Some(constant_collection)) => {
            Some(LastBlockValue::ConstantCollection(constant_collection))
        }
        (Some(constant_binding), None) => {
            Some(LastBlockValue::ConstantBinding(Some(constant_binding)))
        }
        (Some(_), Some(_)) => {
            unreachable!("Not possible to evaluate a value both as a constant and a collection")
        }
    }
}

fn get_local_id(definition: &Definition) -> Option<LocalId> {
    match definition {
        Definition::Local(local_id) => Some(*local_id),
        _ => None,
    }
}

#[derive(Debug)]
struct ConstScope {
    scopes: Vec<HashMap<LocalId, SignedField>>,
    scopes_for_collections: Vec<HashMap<LocalId, Vec<Option<SignedField>>>>,
    last_block_value: Option<LastBlockValue>,
}

impl ConstScope {
    fn new() -> Self {
        Self {
            scopes: vec![HashMap::new()],
            scopes_for_collections: vec![HashMap::new()],
            last_block_value: None,
        }
    }

    fn push_scope(&mut self) {
        self.scopes.push(HashMap::new());
        self.scopes_for_collections.push(HashMap::new());
    }

    fn pop_scope(&mut self) {
        self.scopes.pop();
        self.scopes_for_collections.pop();
    }

    fn insert(&mut self, id: LocalId, value: SignedField) {
        if let Some(scope) = self.scopes.last_mut() {
            scope.insert(id, value);
        }
    }

    fn insert_collection(&mut self, id: LocalId, value: Vec<Option<SignedField>>) {
        if let Some(scope_for_collection) = self.scopes_for_collections.last_mut() {
            scope_for_collection.insert(id, value);
        }
    }

    /// Inserts the given last block value. If there was a previous last block value
    /// it returns it.
    fn insert_last_block_value(
        &mut self,
        last_block_value: Option<LastBlockValue>,
    ) -> Option<LastBlockValue> {
        std::mem::replace(&mut self.last_block_value, last_block_value)
    }

    fn get(&self, id: &LocalId) -> Option<SignedField> {
        self.scopes.iter().rev().filter_map(|local_scope| local_scope.get(id).copied()).next()
    }

    fn get_collection(&self, id: &LocalId) -> Option<&Vec<Option<SignedField>>> {
        self.scopes_for_collections
            .iter()
            .rev()
            .filter_map(|local_scope| local_scope.get(id))
            .next()
    }

    fn get_last_block_value(&self) -> Option<&LastBlockValue> {
        self.last_block_value.as_ref()
    }

    #[allow(dead_code)]
    fn consume_last_block_value(&mut self) -> Option<LastBlockValue> {
        self.last_block_value.take()
    }

    #[allow(dead_code)]
    /// Tries to consume the last block value. Consumes it and returns it if it is of type
    /// constant binding. Otherwise it doesn't consume it.
    fn try_to_consume_constant_binding_last_block_value(&mut self) -> Option<SignedField> {
        match self.last_block_value.take() {
            Some(LastBlockValue::ConstantBinding(constant_binding)) => constant_binding,
            Some(collection_value) => {
                self.last_block_value = Some(collection_value);
                None
            }
            None => None,
        }
    }

    /// Tries to consume the last block value. Consumes it and returns it if it is of type
    /// constant collection. Otherwise it doesn't consume it.
    fn try_to_consume_collection_last_block_value(&mut self) -> Option<Vec<Option<SignedField>>> {
        match self.last_block_value.take() {
            Some(LastBlockValue::ConstantCollection(constant_collection)) => {
                Some(constant_collection)
            }
            Some(constant_binding) => {
                self.last_block_value = Some(constant_binding);
                None
            }
            None => None,
        }
    }
}

#[derive(Debug)]
enum LastBlockValue {
    ConstantBinding(Option<SignedField>),
    ConstantCollection(Vec<Option<SignedField>>),
}
