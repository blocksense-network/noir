//! Because Verus doesn't fully support array mutation we
//! change the AST to have only operations which Verus supports.
//!
//! Verus supports none of the following ones:
//! - `arr[i][j] = expr`
//! - `(arr[i]).1 = expr`
//!
//! We want to support array mutation because arrays
//! are a fundamental part of Noir.
//!
//! What this optimization does:
//! `y[0][1][2].3 = x` =>
//! ```
//! {
//!     let mut y%i0%i1%i2 = y[0][1][2];
//!     y%i0%i1%i2.3 = x;
//!
//!     let mut y%i0%i1 = y[0][1];
//!     y%i0%i1[2] = y%i0%i1%i2;
//!
//!     let mut y%i0 = y[0];
//!     y%i0[1] = y%i0%i1;
//!
//!     y[0] = y%i0;
//! }
//! ```
//!
//! `y[0][1].2.3 = x` =>
//! ```
//! {
//!     let mut y%i0%i1%t2 = y[0][1].2;
//!     y%i0%i1%t2.3 = x;
//!
//!     let mut y%i0%i1 = y[0][1];
//!     y%i0%i1.2 = y%i0%i1%t2;
//!
//!     let mut y%i0 = y[0];
//!     y%i0[1] = y%i0%i1;
//!
//!     y[0] = y%i0;
//! }
//! ```
//!

use std::{borrow::Cow, u32};

use noirc_errors::Location;
use noirc_frontend::monomorphization::ast::{
    Assign, Binary, BinaryOp, Call, Definition, Expression, FuncId, Function, Ident, IdentId,
    Index, LValue, Let, LocalId, Program, Type,
};

use crate::vir::vir_gen::expr_to_vir::expression_location;

pub fn fix_array_mutation_pass(program: &mut Program) {
    program.functions.iter_mut().for_each(fix_array_mutation);
}

// Only fix array mutation in function body because FV attributes'
// bodies don't have assign operations
fn fix_array_mutation(function: &mut Function) {
    fix_assign_expression(&mut function.body);
}

// The only mutation operations which we have to fix are located
// in `Assign` expressions
fn fix_assign_expression(expr: &mut Expression) {
    let assign_expr_location = if matches!(expr, Expression::Assign(_)) {
        // This is not a cheap operation and we only want to do it when we need it.
        // It's not inside of the `match` branch because of the Rust borrow checker.
        expression_location(expr)
    } else {
        None
    };

    match expr {
        Expression::Block(expressions) => expressions.iter_mut().for_each(fix_assign_expression),
        Expression::For(for_expression) => fix_assign_expression(&mut for_expression.block),
        Expression::Loop(expression) => fix_assign_expression(expression),
        Expression::While(while_expression) => fix_assign_expression(&mut while_expression.body),
        Expression::If(if_expression) => {
            fix_assign_expression(&mut if_expression.consequence);
            if let Some(alternative) = if_expression.alternative.as_deref_mut() {
                fix_assign_expression(alternative);
            }
        }
        Expression::Match(match_expression) => {
            match_expression
                .cases
                .iter_mut()
                .for_each(|match_case| fix_assign_expression(&mut match_case.branch));

            if let Some(default_case) = match_expression.default_case.as_deref_mut() {
                fix_assign_expression(default_case);
            }
        }
        Expression::Assign(assign) => {
            if !is_lvalue_valid(&assign.lvalue, false, false) {
                *expr = fix_assign_expression_inner(assign, assign_expr_location);
            }
        }
        Expression::Semi(expression) => fix_assign_expression(expression),
        Expression::Clone(expression) => fix_assign_expression(expression),
        Expression::Drop(expression) => fix_assign_expression(expression),
        _ => (), // Other expressions can't have `Assign` as inner expression
    }
}

/// If we have an "invalid" expression like the following one `y[0][1][2].3 = x;` this function
/// swaps it with a block of valid expressions. For the given example we will generate the following block:
/// ```
/// {
///     let mut y%i0%i1%i2 = y[0][1][2];
///     y%i0%i1%i2.3 = x;
///
///     let mut y%i0%i1 = y[0][1];
///     y%i0%i1[2] = y%i0%i1%i2;
///
///     let mut y%i0 = y[0];
///     y%i0[1] = y%i0%i1;
///
///     y[0] = y%i0;
/// }
/// ```
fn fix_assign_expression_inner(assign_expr: &mut Assign, location: Option<Location>) -> Expression {
    let starting_id_for_gen_vars: u32 = 0;
    // Expressions which are generated and will be turned into a Expression::Block
    let mut generated_expressions: Vec<Expression> = Vec::new();
    let first_lhs_ident: Ident = define_first_lhs_ident(
        &assign_expr.lvalue,
        starting_id_for_gen_vars,
        location,
        assign_expr
            .expression
            .return_type()
            .expect("The type of rhs of assign expression should be known")
            .as_ref(),
    );
    // Turns `y[0][1][2].3` into `y[0][1][2].3` but the new one is of
    // type `Expression` and not `LValue`
    let first_rhs: Expression = lvalue_to_expr(&assign_expr.lvalue);

    let (final_lhs, final_rhs) = generate_let_mut_assign_exprs(
        assign_expr.lvalue.clone(), // Original lhs of assume expression
        first_lhs_ident,
        first_rhs,
        *assign_expr.expression.clone(), // Original rhs of assume expression
        &mut generated_expressions,
        starting_id_for_gen_vars + 1,
        location,
    );
    // Define final assign expression, which assigns `final_lhs` to `final_rhs`.
    let final_assign_expr =
        Expression::Assign(Assign { lvalue: final_lhs, expression: Box::new(final_rhs) });

    generated_expressions.push(final_assign_expr);

    // Call the magic function called `assume()` which will be converted
    // to VIR assume expression.
    // We assume that `y[0][1][2].3 = x` happened successfully.
    let assume_expr: Expression = Expression::Call(Call {
        func: Box::new(Expression::Ident(Ident {
            location: location,
            definition: Definition::Function(FuncId(u32::MAX)),
            mutable: false,
            name: String::from("assume"),
            typ: Type::Unit,
            id: IdentId(u32::MAX),
        })),
        arguments: vec![Expression::Binary(Binary {
            lhs: Box::new(lvalue_to_expr(&assign_expr.lvalue)),
            operator: BinaryOp::Equal,
            rhs: assign_expr.expression.clone(),
            location: location.unwrap_or(Location::dummy()),
        })],
        return_type: Type::Unit,
        location: location.unwrap_or(Location::dummy()),
    });

    generated_expressions.push(assume_expr);

    Expression::Block(generated_expressions)
}

/// For an lhs expression of type `y[0][1][2].3` will build an Ident with
/// name `y%i0%i1%i2%t3`
fn define_first_lhs_ident(
    lvalue: &LValue,
    local_id: u32,
    location: Option<Location>,
    typ: &Type,
) -> Ident {
    Ident {
        location,
        definition: Definition::Local(LocalId(local_id)),
        mutable: true,
        name: build_lvalue_string(lvalue),
        typ: typ.clone(),
        id: IdentId(local_id),
    }
}
/// For an lhs expression of type `y[0][index][2].3` will build a String `y%i0%iindex%i2%t3`
fn build_lvalue_string(lvalue: &LValue) -> String {
    let mut result = String::new();
    build_lvalue_string_recursive(lvalue, &mut result);
    result
}

fn build_lvalue_string_recursive(lvalue: &LValue, result: &mut String) {
    match lvalue {
        LValue::Ident(ident) => {
            result.push_str(&ident.name);
        }
        LValue::Index { array, index, .. } => {
            build_lvalue_string_recursive(array, result);
            result.push_str("%i");
            result.push_str(&index.to_string());
        }
        LValue::MemberAccess { object, field_index } => {
            build_lvalue_string_recursive(object, result);
            result.push_str("%t");
            result.push_str(&field_index.to_string());
        }
        LValue::Dereference { reference, .. } => {
            // We ignore dereference operation
            build_lvalue_string_recursive(reference, result);
        }
    }
}

/// Generate recursively the expressions which will swap the mutation operation.
/// For each iteration we peel off the outer layer of the argument `original_lhs_value`.
/// This is required because after the recursive generation of code we assign the remaining
/// rhs expression to what has left of the original lhs expression.
fn generate_let_mut_assign_exprs(
    // We peel the outer layer of the LValue in each call until it is a valid LValue
    original_lhs_lvalue: LValue,
    last_lhs_as_ident: Ident,
    last_rhs: Expression,
    next_rhs_expr: Expression,
    generated_expressions: &mut Vec<Expression>,
    last_local_id: u32,
    location_of_original_assign_expr: Option<Location>,
) -> (LValue, Expression) {
    if is_lvalue_valid(&original_lhs_lvalue, false, false) {
        (original_lhs_lvalue, Expression::Ident(last_lhs_as_ident))
    } else {
        // If we have a rhs expression like `y[0][1][2]` we want to "remove" the outer layer
        // We will get the outer expression which is `Index(y[0][1], 2)` and the new_rhs is `y[0][1]`
        let (outer_expr, new_rhs) = remove_outer_expression(last_rhs);
        // Get the new name for the lhs which we will `Let` define
        // E.g. `y%i0%ik%i2%t3` -> `y%i0%ik%i2`
        let new_lhs_name = pop_last_operation_from_name(last_lhs_as_ident.name.clone());
        let new_lhs_type_ref = match new_rhs.return_type().expect("Expected a known type") {
            Cow::Borrowed(b) => b.clone(),
            Cow::Owned(o) => o.clone(),
        };

        let let_expression = Expression::Let(Let {
            id: LocalId(last_local_id + 1),
            mutable: true,
            name: new_lhs_name.clone(),
            expression: Box::new(new_rhs.clone()),
        });

        generated_expressions.push(let_expression);

        // We use the ident for the rhs of the next assign expression
        let ident_of_let_expression = Ident {
            location: location_of_original_assign_expr,
            definition: Definition::Local(LocalId(last_local_id + 1)),
            mutable: true,
            name: new_lhs_name,
            typ: new_lhs_type_ref.clone(),
            id: IdentId(last_local_id + 1),
        };
        let assign_expr = Expression::Assign(Assign {
            lvalue: try_index_op_expr_to_lvalue(
                outer_expr.expect(
                    "If lhs LValue is still invalid we should be able to \
                    strip an outer expression from the last rhs",
                ),
                ident_of_let_expression.clone(),
            )
            .expect("Should be convertable"),
            expression: Box::new(next_rhs_expr),
        });

        generated_expressions.push(assign_expr);

        generate_let_mut_assign_exprs(
            remove_outer_lvalue(original_lhs_lvalue),
            ident_of_let_expression.clone(),
            new_rhs,
            Expression::Ident(ident_of_let_expression),
            generated_expressions,
            last_local_id + 1,
            location_of_original_assign_expr,
        )
    }
}

/// Тakes a string like `y%i0%ik%i2%t3` and returns `y%i0%ik%i2`.
fn pop_last_operation_from_name(name: String) -> String {
    let mut last_op_pos = None;
    let mut chars = name.chars().collect::<Vec<_>>();

    let mut i = chars.len();
    while i >= 2 {
        i -= 1;
        if chars[i - 1] == '%' && (chars[i] == 'i' || chars[i] == 't') {
            last_op_pos = Some(i - 1);
            break;
        }
    }

    match last_op_pos {
        Some(pos) => {
            chars.truncate(pos);
            chars.into_iter().collect()
        }
        None => name,
    }
}

fn lvalue_to_expr(lvalue: &LValue) -> Expression {
    match lvalue {
        LValue::Ident(ident) => Expression::Ident(ident.clone()),
        LValue::Index { array, index, element_type, location } => Expression::Index(Index {
            collection: Box::new(lvalue_to_expr(&array)),
            index: index.clone(),
            element_type: element_type.clone(),
            location: *location,
        }),
        LValue::MemberAccess { object, field_index } => {
            Expression::ExtractTupleField(Box::new(lvalue_to_expr(object)), *field_index)
        }
        LValue::Dereference { reference, .. } => {
            //TODO(totel): We are currently ignoring dereference operation
            lvalue_to_expr(reference)
        }
    }
}

/// Removes most outer operation if it is `Index` or `ExtractTupleField`.
/// Returns the removed expression if there was such an expression.
/// Returns the new modified expression
fn remove_outer_expression(expr: Expression) -> (Option<Expression>, Expression) {
    match expr {
        Expression::Index(index) => {
            let inner_expr = *index.collection.clone();
            (Some(Expression::Index(index)), inner_expr)
        }
        Expression::ExtractTupleField(inner, idx) => {
            let inner_expr = *inner.clone();
            (Some(Expression::ExtractTupleField(inner, idx)), inner_expr)
        }
        _ => (None, expr),
    }
}

/// Tries to convert Expression to LValue. If unsuccessful returns `None`
/// This functions expects an operation of type `Expression::Index` or `Expression::ExtractTupleField`.
/// Converts those expression to their corresponding LValue ops with arg `ident`.
fn try_index_op_expr_to_lvalue(expr: Expression, ident: Ident) -> Option<LValue> {
    match expr {
        Expression::Index(index) => Some(LValue::Index {
            array: Box::new(LValue::Ident(ident)),
            index: index.index,
            element_type: index.element_type,
            location: index.location,
        }),
        Expression::ExtractTupleField(_, tuple_index) => Some(LValue::MemberAccess {
            object: Box::new(LValue::Ident(ident)),
            field_index: tuple_index,
        }),
        _ => None,
    }
}

/// This method acts as an identity function if `lvalue` is of type `Ident`.
fn remove_outer_lvalue(lvalue: LValue) -> LValue {
    match lvalue {
        LValue::Ident(_) => lvalue,
        LValue::Index { array, .. } => *array,
        LValue::MemberAccess { object, .. } => *object,
        LValue::Dereference { reference, .. } => *reference,
    }
}

/// If we have a lvalue of type `Index(...(Index(...)))` we call this lvalue invalid.
/// If we have a lvalue of type `MemberAccess(...(Index(...)))` we call this lvalue invalid.
/// In any other case the lvalue is valid.
fn is_lvalue_valid(lvalue: &LValue, is_inside_index: bool, is_inside_member_access: bool) -> bool {
    match lvalue {
        LValue::Ident(_) => true,
        LValue::Index { array, .. } => {
            !is_inside_index
                && !is_inside_member_access
                && is_lvalue_valid(&array, true, is_inside_member_access)
        }
        LValue::MemberAccess { object, .. } => is_lvalue_valid(&object, is_inside_index, true),
        LValue::Dereference { reference, .. } => {
            is_lvalue_valid(&reference, is_inside_index, is_inside_member_access)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pop_last_operation() {
        assert_eq!(pop_last_operation_from_name("y%i0%ik%i2%t3".to_string()), "y%i0%ik%i2");
        assert_eq!(pop_last_operation_from_name("y%i4%i%result".to_string()), "y%i4");
        assert_eq!(pop_last_operation_from_name("x".to_string()), "x");
        assert_eq!(pop_last_operation_from_name("x%t0".to_string()), "x");
        assert_eq!(pop_last_operation_from_name("x%i0%t1".to_string()), "x%i0");
        assert_eq!(pop_last_operation_from_name("x%i%y".to_string()), "x");
    }
}
