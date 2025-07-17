//! Verus performs a transformation on function parameters where mutable arguments are
//! converted into immutable ones, and a `let mut` binding is inserted at the beginning
//! of the function body. To stay consistent with this behavior, we apply the same
//! transformation during our monomorphized AST passes.
//!
//! Specifically, the function:
//! ```
//! fn foo(mut x: T) {
//!     ...
//! }
//! ```
//! is transformed into:
//! ```
//! fn foo(x: T) {
//!     let mut x = x;
//!     ...
//! }
//! ```
//!
//! In short, `mut` is removed from the parameter list, and a mutable shadowing
//! `let` binding is inserted at the top of the function body.

use noirc_errors::Location;
use noirc_frontend::{
    hir_def::stmt::HirPattern,
    monomorphization::ast::{Definition, Expression, Function, Ident, IdentId, Let, Program},
};

pub fn demut_parameters(program: &mut Program) {
    program.functions.iter_mut().for_each(demut_parameters_inner);
}

fn demut_parameters_inner(function: &mut Function) {
    insert_let_mut_exprs(function);
    convert_mut_params_to_non_mut(function)
}

fn insert_let_mut_exprs(function: &mut Function) {
    let mut_params_locations: Vec<Option<Location>> = function
        .func_sig
        .0
        .iter()
        .map(|(hir_pattern, ..)| match hir_pattern {
            HirPattern::Mutable(_, location) => Some(*location),
            _ => None,
        })
        .collect();

    function
        .parameters
        .iter()
        .zip(mut_params_locations)
        .filter(|(param, _)| param.1) // is mut
        // Get parameter's local id, name, type and location
        .map(|(param, location)| (param.0, param.2.clone(), param.3.clone(), location))
        .rev()
        .for_each(|(local_id, param_name, param_type, location)| {
            // Insert `let` expression at the start of the body's block
            if let Expression::Block(block) = &mut function.body {
                block.insert(
                    0,
                    Expression::Let(Let {
                        id: local_id,
                        mutable: true,
                        name: param_name.clone(),
                        expression: Box::new(Expression::Ident(Ident {
                            location,
                            definition: Definition::Local(local_id),
                            mutable: false,
                            name: param_name,
                            typ: param_type,
                            id: IdentId(local_id.0),
                        })),
                    }),
                );
            }
        });
}

fn convert_mut_params_to_non_mut(function: &mut Function) {
    function.parameters.iter_mut().for_each(|param| {
        param.1 = false; // Set mut to false 
    });

    function.func_sig.0.iter_mut().for_each(|param| {
        if let HirPattern::Mutable(ref inner, _) = param.0 {
            // Clone gets optimized away
            param.0 = *inner.clone()
        }
    });
}
