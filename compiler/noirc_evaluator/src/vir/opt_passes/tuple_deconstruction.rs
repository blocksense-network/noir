use noirc_frontend::monomorphization::ast::{
    Expression, Function, Ident, Let, Program, Type,
};

pub fn fix_tuple_deconstruction_pass(program: &mut Program) {
    program.functions.iter_mut().for_each(|function| fix_tuple_deconstruction(function));
}

fn fix_tuple_deconstruction(function: &mut Function) {
    fix_tuple_deconstruction_expression(&mut function.body);
}

/// If the given `expression` is a block where:
/// - The first expression is itself a block, and
/// - The first statement inside that inner block is a `let _ = (tuple)`
///
/// Then:
/// - This function rewrites the inner block, and
/// - Afterwards flattens it into the outer block
fn fix_tuple_deconstruction_expression(expression: &mut Expression) {
    if let Expression::Block(outer_block) = expression {
        let mut blocks_to_be_flattened: Vec<(usize, Vec<Expression>)> = Vec::new();

        // NOTE: enumeration done beforehand to keep real original indices
        outer_block
            .iter_mut()
            .enumerate()
            .filter_map(|(i, expression)| {
                if let Expression::Block(inner_expressions) = expression {
                    Some((i, inner_expressions))
                } else {
                    None
                }
            })
            .for_each(|(i, expressions)| {
                let let_exprs_for_deconstruct_tuple: Option<usize> = expressions
                    .first()
                    .and_then(|first_expression| match first_expression {
                        Expression::Let(Let { name, expression, .. }) => Some((name, expression)),
                        _ => None,
                    })
                    .and_then(
                        |(name, expression)| if *name == "_" { Some(expression) } else { None },
                    )
                    .and_then(|expression| {
                        let ty = expression.return_type()?;
                        match ty.as_ref() {
                            Type::Tuple(tuple_vec) => Some(tuple_vec.len()),
                            _ => None,
                        }
                    });

                if let Some(exprs_len) = let_exprs_for_deconstruct_tuple {
                    for expr in expressions[1..=exprs_len].iter_mut() {
                        assert!(matches!(expr, Expression::Let(..)), 
                            "Expected to have a follow up auto generated let expressions after tuple deconstruct assignment");
                        if let Expression::Let(let_expr) = expr {
                            fix_rhs_tuple_indexing(let_expr);
                        }
                    }
                    blocks_to_be_flattened.push((i, expressions.clone()));
                }
            });

        // Flatten marked blocks into the outer block
        let mut offset = 0;
        blocks_to_be_flattened.into_iter().for_each(|(original_index, expressions)| {
            let added_expressions_count = expressions.len();
            outer_block.splice(original_index + offset..original_index + offset + 1, expressions);
            offset = offset + added_expressions_count - 1;
        });
    }
}

fn fix_rhs_tuple_indexing(expr: &mut Let) {
    if let Expression::ExtractTupleField(rhs_ident, _) = expr.expression.as_mut() {
        if let Expression::Ident(Ident { name, .. }) = rhs_ident.as_mut() {
            *name = String::from("_");
        }
    }
}
