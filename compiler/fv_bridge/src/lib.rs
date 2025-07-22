use std::collections::{BTreeMap, HashMap};

use fm::FileId;
use formal_verification::parse::parse_attribute;
use iter_extended::vecmap;
use noirc_driver::{CompilationResult, CompileError, CompileOptions, check_crate};
use noirc_errors::{CustomDiagnostic, Location};
use noirc_evaluator::vir::vir_gen::Attribute;
use noirc_evaluator::{
    errors::{RuntimeError, SsaReport},
    vir::{create_verus_vir_with_ready_annotations, vir_gen::BuildingKrateError},
};
use noirc_frontend::hir::resolution::errors::ResolverError;
use noirc_frontend::hir_def::expr::{HirCallExpression, HirExpression, HirLiteral};
use noirc_frontend::node_interner::{DefinitionKind, ExprId};
use noirc_frontend::{Kind, TypeVariable};
// use noirc_frontend::node_interner::FuncId;
use noirc_frontend::{
    debug::DebugInstrumenter,
    graph::CrateId,
    hir::Context,
    monomorphization::{
        Monomorphizer,
        ast::{FuncId, Program},
        debug_types::DebugTypeTracker,
        errors::MonomorphizationError,
        perform_impl_bindings, perform_instantiation_bindings, undo_instantiation_bindings,
    },
    node_interner::{self, NodeInterner},
    parser::ParserError,
    token::SecondaryAttributeKind,
};
use vir::ast::Krate;

pub fn compile_and_build_vir_krate(
    context: &mut Context,
    crate_id: CrateId,
    options: &CompileOptions,
) -> CompilationResult<Krate> {
    modified_compile_main(context, crate_id, options)
}

fn modified_compile_main(
    context: &mut Context,
    crate_id: CrateId,
    options: &CompileOptions,
) -> CompilationResult<Krate> {
    let (_, mut warnings) = check_crate(context, crate_id, options)?;

    let main = context.get_main_function(&crate_id).ok_or_else(|| {
        let err = CustomDiagnostic::from_message(
            "cannot compile crate into a program as it does not contain a `main` function",
            FileId::default(),
        );
        vec![err]
    })?;

    let compiled_program =
        modified_compile_no_check(context, options, main).map_err(|error| match error {
            CompileOrResolverError::CompileError(compile_error) => vec![compile_error.into()],
            CompileOrResolverError::ResolverErrors(resolver_errors) => {
                resolver_errors.iter().map(Into::into).collect()
            }
        })?;

    let compilation_warnings = vecmap(compiled_program.warnings.clone(), CustomDiagnostic::from);
    if options.deny_warnings && !compilation_warnings.is_empty() {
        return Err(compilation_warnings);
    }
    if !options.silence_warnings {
        warnings.extend(compilation_warnings);
    }

    Ok((compiled_program.krate, warnings))
}

pub enum CompileOrResolverError {
    CompileError(CompileError),
    ResolverErrors(Vec<ResolverError>),
}

// Something like the method `compile_no_check()`
fn modified_compile_no_check(
    context: &mut Context,
    options: &CompileOptions,
    main_function: node_interner::FuncId,
) -> Result<KrateAndWarnings, CompileOrResolverError> {
    let force_unconstrained = options.force_brillig || options.minimal_ssa;

    // println!("Node interner: {:#?}", context.def_interner);

    let (program, fv_annotations) = modified_monomorphize(
        main_function,
        &mut context.def_interner,
        &DebugInstrumenter::default(),
        force_unconstrained,
    )
    .map_err(|e| match e {
        MonomorphOrResolverError::MonomorphizationError(monomorphization_error) => {
            CompileOrResolverError::CompileError(CompileError::MonomorphizationError(
                monomorphization_error,
            ))
        }
        MonomorphOrResolverError::ResolverErrors(resolver_errors) => {
            CompileOrResolverError::ResolverErrors(resolver_errors)
        }
    })?;

    if options.show_monomorphized {
        println!("{program}");
    }

    Ok(KrateAndWarnings {
        krate: create_verus_vir_with_ready_annotations(program, fv_annotations)
            .map_err(|BuildingKrateError::Error(msg)| {
                RuntimeError::InternalError(noirc_evaluator::errors::InternalError::General {
                    message: msg,
                    call_stack: vec![],
                })
            })
            .map_err(|runtime_error| {
                CompileOrResolverError::CompileError(CompileError::RuntimeError(runtime_error))
            })?,
        warnings: vec![],
        parse_annotations_errors: vec![], // TODO(totel): Get the errors from `modified_monomorphize()`
    })
}

enum MonomorphOrResolverError {
    MonomorphizationError(MonomorphizationError),
    ResolverErrors(Vec<ResolverError>),
}

fn modified_monomorphize(
    main: node_interner::FuncId,
    interner: &mut NodeInterner,
    debug_instrumenter: &DebugInstrumenter,
    force_unconstrained: bool,
) -> Result<(Program, Vec<(FuncId, Vec<Attribute>)>), MonomorphOrResolverError> {
    let debug_type_tracker = DebugTypeTracker::build_from_debug_instrumenter(debug_instrumenter);
    // TODO(totel): Monomorphizer is a `pub(crate)` struct
    let mut monomorphizer = Monomorphizer::new(interner, debug_type_tracker);
    monomorphizer.in_unconstrained_function = force_unconstrained;
    let function_sig = monomorphizer
        .compile_main(main)
        .map_err(MonomorphOrResolverError::MonomorphizationError)?;
    let mut new_ids_to_old_ids: HashMap<FuncId, node_interner::FuncId> = HashMap::new();
    new_ids_to_old_ids.insert(Program::main_id(), main);

    while !monomorphizer.queue.is_empty() {
        let (next_fn_id, new_id, bindings, trait_method, is_unconstrained, location) =
            monomorphizer.queue.pop_front().unwrap();
        monomorphizer.locals.clear();

        monomorphizer.in_unconstrained_function = is_unconstrained;

        perform_instantiation_bindings(&bindings);
        let interner = &monomorphizer.interner;
        let impl_bindings = perform_impl_bindings(interner, trait_method, next_fn_id, location)
            .map_err(MonomorphizationError::InterpreterError)
            .map_err(MonomorphOrResolverError::MonomorphizationError)?;

        monomorphizer
            .function(next_fn_id, new_id, location)
            .map_err(MonomorphOrResolverError::MonomorphizationError)?;
        new_ids_to_old_ids.insert(new_id, next_fn_id);
        undo_instantiation_bindings(impl_bindings);
        undo_instantiation_bindings(bindings);
    }
    // ________________________My hack snippet___________________________________

    let func_id = monomorphizer.interner.find_function("generic_func").unwrap();
    let func_id_as_expr_id = monomorphizer.interner.function(&func_id).as_expr();

    let pseudo_arg = monomorphizer.interner.push_expr_full(
        HirExpression::Literal(HirLiteral::Bool(true)),
        Location::dummy(),
        noirc_frontend::Type::Bool,
    );

    let pseudo_call_expr = HirExpression::Call(HirCallExpression {
        func: func_id_as_expr_id,
        arguments: vec![pseudo_arg],
        location: Location::dummy(),
        is_macro_call: false,
    });

    let pseudo_call_expr_id = monomorphizer.interner.push_expr_full(
        pseudo_call_expr,
        Location::dummy(),
        noirc_frontend::Type::Unit,
    );

    // panic!("{:#?}", noirc_frontend::Type::Bool.instantiate(&monomorphizer.interner));

    // panic!("TEST: {:#?}", monomorphizer.interner.expression(&func_id_as_expr_id));
    // let arg_type_var_id =
    // monomorphizer.interner.function_meta(&func_id).direct_generics[0].type_var.id();

    let mut typ_bindings = noirc_frontend::Type::Unit.instantiate(&monomorphizer.interner).1;
    // let type_var_id = arg_type_var_id; // monomorphizer.interner.next_type_variable_id();

    let parameter_index = 0;
    let parameter_type =
        &monomorphizer.interner.function_meta(&func_id).parameters.0[parameter_index].1;
    let type_variable = if let noirc_frontend::Type::NamedGeneric(named_generic) = &parameter_type {
        &named_generic.type_var
    } else {
        panic!("{}", parameter_type)
    };
    // THE LAST ARG IN THE FOLLOWING `.insert` IS THE IMPORTANT ONE
    typ_bindings.insert(
        type_variable.id(),
        (type_variable.clone(), Kind::Normal, noirc_frontend::Type::Bool),
    );

    monomorphizer.interner.store_instantiation_bindings(pseudo_call_expr_id, typ_bindings);

    monomorphizer.queue_function(
        func_id,
        pseudo_call_expr_id,
        monomorphizer.interner.id_type(func_id_as_expr_id),
        vec![],
        None,
    );

    while !monomorphizer.queue.is_empty() {
        let (next_fn_id, new_id, bindings, trait_method, is_unconstrained, location) =
            monomorphizer.queue.pop_front().unwrap();
        monomorphizer.locals.clear();

        monomorphizer.in_unconstrained_function = is_unconstrained;

        perform_instantiation_bindings(&bindings);
        let interner = &monomorphizer.interner;
        let impl_bindings = perform_impl_bindings(interner, trait_method, next_fn_id, location)
            .map_err(MonomorphizationError::InterpreterError)
            .map_err(MonomorphOrResolverError::MonomorphizationError)?;

        monomorphizer
            .function(next_fn_id, new_id, location)
            .map_err(MonomorphOrResolverError::MonomorphizationError)?;
        new_ids_to_old_ids.insert(new_id, next_fn_id);
        undo_instantiation_bindings(impl_bindings);
        undo_instantiation_bindings(bindings);
    }

    // ________________________End of hack snippet___________________________________

    let func_sigs = monomorphizer
        .finished_functions
        .iter()
        .flat_map(|(_, f)| {
            if (!force_unconstrained && f.inline_type.is_entry_point())
                || f.id == Program::main_id()
            {
                Some(f.func_sig.clone())
            } else {
                None
            }
        })
        .collect();

    let globals = monomorphizer.finished_globals.into_iter().collect::<BTreeMap<_, _>>();

    let fv_annotations: Vec<(FuncId, Vec<Attribute>)> = monomorphizer
        .finished_functions
        .iter()
        .filter_map(|(new_func_id, function)| {
            new_ids_to_old_ids.get(new_func_id).map(|old_id| (new_func_id, old_id, function))
        })
        .map(|(new_func_id, old_id, function)| {
            let tag_attributes: Vec<(&str, Location)> = monomorphizer
                .interner
                .function_attributes(old_id)
                .secondary
                .iter()
                .filter_map(|attribute| match &attribute.kind {
                    SecondaryAttributeKind::Tag(annotation) => {
                        Some((annotation.as_str(), attribute.location))
                    }
                    _ => None,
                })
                .collect();

            Ok((
                new_func_id.clone(),
                tag_attributes
                    .into_iter()
                    .map(|(annotation_body, location)| {
                        parse_attribute(
                            annotation_body,
                            location,
                            function,
                            &globals,
                            &monomorphizer.finished_functions,
                        )
                        .map_err(MonomorphOrResolverError::ResolverErrors)
                    })
                    .collect::<Result<Vec<_>, _>>()?,
            ))
        })
        .collect::<Result<Vec<_>, _>>()?;

    let functions = vecmap(monomorphizer.finished_functions, |(_, f)| f);

    let (debug_variables, debug_functions, debug_types) =
        monomorphizer.debug_type_tracker.extract_vars_and_types();

    let program = Program::new(
        functions,
        func_sigs,
        function_sig,
        monomorphizer.return_location,
        globals,
        debug_variables,
        debug_functions,
        debug_types,
    );

    Ok((program.handle_ownership(), fv_annotations))
}

pub struct KrateAndWarnings {
    pub krate: Krate,
    pub warnings: Vec<SsaReport>,
    pub parse_annotations_errors: Vec<ParserError>,
}

// pub fn monomorphize_one_function() {
//     let func_id = monomorphizer.interner.find_function("generic_func").unwrap();
//     let func_id_as_expr_id = monomorphizer.interner.function(&func_id).as_expr();

//     let pseudo_arg = monomorphizer.interner.push_expr_full(
//         HirExpression::Literal(HirLiteral::Bool(true)),
//         Location::dummy(),
//         noirc_frontend::Type::Bool,
//     );

//     let pseudo_call_expr = HirExpression::Call(HirCallExpression {
//         func: func_id_as_expr_id,
//         arguments: vec![pseudo_arg],
//         location: Location::dummy(),
//         is_macro_call: false,
//     });

//     let pseudo_call_expr_id = monomorphizer.interner.push_expr_full(
//         pseudo_call_expr,
//         Location::dummy(),
//         noirc_frontend::Type::Unit,
//     );


//     let mut typ_bindings = noirc_frontend::Type::Unit.instantiate(&monomorphizer.interner).1;

//     let parameter_index = 0;
//     let parameter_type =
//         &monomorphizer.interner.function_meta(&func_id).parameters.0[parameter_index].1;
//     let type_variable = if let noirc_frontend::Type::NamedGeneric(named_generic) = &parameter_type {
//         &named_generic.type_var
//     } else {
//         panic!("{}", parameter_type)
//     };
//     // THE LAST ARG IN THE FOLLOWING `.insert` IS THE IMPORTANT ONE
//     typ_bindings.insert(
//         type_variable.id(),
//         (type_variable.clone(), Kind::Normal, noirc_frontend::Type::Bool),
//     );

//     monomorphizer.interner.store_instantiation_bindings(pseudo_call_expr_id, typ_bindings);

//     monomorphizer.queue_function(
//         func_id,
//         pseudo_call_expr_id,
//         monomorphizer.interner.id_type(func_id_as_expr_id),
//         vec![],
//         None,
//     );

//     while !monomorphizer.queue.is_empty() {
//         let (next_fn_id, new_id, bindings, trait_method, is_unconstrained, location) =
//             monomorphizer.queue.pop_front().unwrap();
//         monomorphizer.locals.clear();

//         monomorphizer.in_unconstrained_function = is_unconstrained;

//         perform_instantiation_bindings(&bindings);
//         let interner = &monomorphizer.interner;
//         let impl_bindings = perform_impl_bindings(interner, trait_method, next_fn_id, location)
//             .map_err(MonomorphizationError::InterpreterError)
//             .map_err(MonomorphOrResolverError::MonomorphizationError)?;

//         monomorphizer
//             .function(next_fn_id, new_id, location)
//             .map_err(MonomorphOrResolverError::MonomorphizationError)?;
//         new_ids_to_old_ids.insert(new_id, next_fn_id);
//         undo_instantiation_bindings(impl_bindings);
//         undo_instantiation_bindings(bindings);
//     }

//     todo!()
// }
