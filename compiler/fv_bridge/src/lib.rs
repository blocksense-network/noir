use fm::FileId;
use formal_verification::ast::{AnnExpr, SpannedTypedExpr};
use formal_verification::convert_structs::convert_struct_access_to_tuple_access;
use formal_verification::type_conversion::convert_mast_to_noir_type;
use formal_verification::typing::{OptionalType, TypeInferenceError, type_infer};
use formal_verification::{State, parse::parse_attribute};
use iter_extended::vecmap;
use noirc_driver::{CompilationResult, CompileError, CompileOptions, check_crate};
use noirc_errors::Location;
use noirc_errors::{CustomDiagnostic, Span};
use noirc_evaluator::vir::vir_gen::Attribute;
use noirc_evaluator::vir::vir_gen::expr_to_vir::std_functions::OLD;
use noirc_evaluator::{
    errors::{RuntimeError, SsaReport},
    vir::{create_verus_vir_with_ready_annotations, vir_gen::BuildingKrateError},
};
use noirc_frontend::Kind;
use noirc_frontend::hir_def::expr::HirCallExpression;
use noirc_frontend::hir_def::expr::{HirExpression, HirLiteral};
use noirc_frontend::hir_def::stmt::HirPattern;
use noirc_frontend::monomorphization::ast::LocalId;
use noirc_frontend::node_interner::{ExprId, FunctionModifiers};
use noirc_frontend::token::SecondaryAttribute;
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
use std::cell::RefCell;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::rc::Rc;
use vir::ast::Krate;

use crate::errors::{CompilationErrorBundle, MonomorphizationErrorBundle};
use crate::typed_attrs_to_vir::convert_typed_attribute_to_vir_attribute;

mod errors;
pub mod typed_attrs_to_vir;

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
            CompilationErrorBundle::CompileError(compile_error) => vec![compile_error.into()],
            CompilationErrorBundle::FvError(fv_monomorphization_error) => {
                vec![fv_monomorphization_error.into()]
            }
            CompilationErrorBundle::TypeError(type_check_error) => {
                vec![CustomDiagnostic::from(&type_check_error)]
            }
            CompilationErrorBundle::ParserErrors(parser_errors) => {
                parser_errors.into_iter().map(CustomDiagnostic::from).collect()
            }
            CompilationErrorBundle::ResolverError(resolver_error) => {
                vec![CustomDiagnostic::from(&resolver_error)]
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

// Something like the method `compile_no_check()`
fn modified_compile_no_check(
    context: &mut Context,
    options: &CompileOptions,
    main_function: node_interner::FuncId,
) -> Result<KrateAndWarnings, CompilationErrorBundle> {
    let force_unconstrained = options.force_brillig || options.minimal_ssa;

    let (program, fv_annotations) = modified_monomorphize(
        main_function,
        &mut context.def_interner,
        &DebugInstrumenter::default(),
        force_unconstrained,
    )
    .map_err(|e| match e {
        MonomorphizationErrorBundle::MonomorphizationError(monomorphization_error) => {
            CompilationErrorBundle::CompileError(CompileError::MonomorphizationError(
                monomorphization_error,
            ))
        }
        MonomorphizationErrorBundle::FvError(fv_monomorphization_error) => {
            CompilationErrorBundle::FvError(fv_monomorphization_error)
        }
        MonomorphizationErrorBundle::TypeError(type_check_error) => {
            CompilationErrorBundle::TypeError(type_check_error)
        }
        MonomorphizationErrorBundle::ParserErrors(parser_errors) => {
            CompilationErrorBundle::ParserErrors(parser_errors)
        }
        MonomorphizationErrorBundle::ResolverError(resolver_error) => {
            CompilationErrorBundle::ResolverError(resolver_error)
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
                CompilationErrorBundle::CompileError(CompileError::RuntimeError(runtime_error))
            })?,
        warnings: vec![],
        parse_annotations_errors: vec![], // TODO(totel): Get the errors from `modified_monomorphize()`
    })
}

enum TypedAttribute {
    Ghost,
    Requires(SpannedTypedExpr),
    Ensures(SpannedTypedExpr),
}

fn modified_monomorphize(
    main: node_interner::FuncId,
    interner: &mut NodeInterner,
    debug_instrumenter: &DebugInstrumenter,
    force_unconstrained: bool,
) -> Result<(Program, Vec<(FuncId, Vec<Attribute>)>), MonomorphizationErrorBundle> {
    let debug_type_tracker = DebugTypeTracker::build_from_debug_instrumenter(debug_instrumenter);
    // NOTE: Monomorphizer is a `pub(crate)` struct which we changed to pub
    let mut monomorphizer = Monomorphizer::new(interner, debug_type_tracker);
    monomorphizer.in_unconstrained_function = force_unconstrained;
    let function_sig = monomorphizer
        .compile_main(main)
        .map_err(MonomorphizationErrorBundle::MonomorphizationError)?;
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
            .map_err(MonomorphizationErrorBundle::MonomorphizationError)?;

        monomorphizer
            .function(next_fn_id, new_id, location)
            .map_err(MonomorphizationErrorBundle::MonomorphizationError)?;
        new_ids_to_old_ids.insert(new_id, next_fn_id);
        undo_instantiation_bindings(impl_bindings);
        undo_instantiation_bindings(bindings);
    }

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

    // Clone because of the borrow checker
    let globals = monomorphizer.finished_globals.clone().into_iter().collect::<BTreeMap<_, _>>();

    let min_available_id: u32 =
        monomorphizer.locals.values().map(|LocalId(id)| *id).max().unwrap_or_default() + 1;
    let min_available_id = Rc::new(RefCell::new(min_available_id));

    let functions_to_process: Vec<(FuncId, node_interner::FuncId)> = monomorphizer
        .finished_functions
        .keys()
        .rev()
        .copied()
        .filter_map(|new_func_id| {
            new_ids_to_old_ids.get(&new_func_id).map(|old_id| (new_func_id, *old_id))
        })
        .collect();

    let mut fv_annotations = Vec::with_capacity(functions_to_process.len());
    // Functions which get resolved via a MonomorphRequest we have to
    // manually add ghost attributes to them.
    let mut to_be_added_ghost_attribute: HashSet<FuncId> = HashSet::new();

    for (new_func_id, old_id) in functions_to_process {
        let parameters_hir_types: HashMap<String, noirc_frontend::Type> =
            collect_parameter_hir_types(&monomorphizer, &old_id);

        let attribute_data: Vec<_> = monomorphizer
            .interner
            .function_attributes(&old_id)
            .secondary
            .iter()
            .filter_map(|attribute| {
                if let SecondaryAttributeKind::Tag(annotation) = &attribute.kind {
                    Some((annotation.as_str().to_owned(), attribute.location))
                } else {
                    None
                }
            })
            .collect();

        let mut processed_attributes = Vec::new();

        for (annotation_body, location) in attribute_data {
            let function_for_parser = &monomorphizer.finished_functions[&new_func_id];

            // NOTE: #['...]
            //       ^^^^^^^ - received `Location`
            //          ^^^  - relevant stuff
            let location = Location {
                span: Span::inclusive(location.span.start() + 3, location.span.end() - 1),
                ..location
            };

            let parsed_attribute = parse_attribute(
                &annotation_body,
                location,
                function_for_parser,
                &globals,
                &monomorphizer.finished_functions,
            )
            .map_err(|e| MonomorphizationErrorBundle::ParserErrors(e.0))?;
            // Ghost functions always get a monomorphization request because
            // they are not part of the monomorphizer.finished_functions
            let typed_attribute = match parsed_attribute {
                formal_verification::Attribute::Ghost => TypedAttribute::Ghost,
                formal_verification::Attribute::Ensures(expr) => {
                    let typed_expr = type_infer_attribute_expr(
                        &mut monomorphizer,
                        new_func_id,
                        &globals,
                        min_available_id.clone(),
                        &parameters_hir_types,
                        expr,
                        &mut new_ids_to_old_ids,
                        &mut to_be_added_ghost_attribute,
                    )?;
                    TypedAttribute::Ensures(typed_expr)
                }
                formal_verification::Attribute::Requires(expr) => {
                    let typed_expr = type_infer_attribute_expr(
                        &mut monomorphizer,
                        new_func_id,
                        &globals,
                        min_available_id.clone(),
                        &parameters_hir_types,
                        expr,
                        &mut new_ids_to_old_ids,
                        &mut to_be_added_ghost_attribute,
                    )?;
                    TypedAttribute::Requires(typed_expr)
                }
            };

            let final_state = State {
                function: &monomorphizer.finished_functions[&new_func_id],
                global_constants: &globals,
                functions: &monomorphizer.finished_functions,
                min_local_id: min_available_id.clone(),
            };

            processed_attributes
                .push(convert_typed_attribute_to_vir_attribute(typed_attribute, &final_state));
        }

        fv_annotations.push((new_func_id, processed_attributes));
    }

    to_be_added_ghost_attribute.into_iter().for_each(|func_id| {
        fv_annotations.push((func_id, vec![Attribute::Ghost]));
    });

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

// Helper function using a bounded for-loop for safer retries.
/// Does type inferring and processes monomorphization requests.
/// Returns the typed attribute expression and a flag if monomorphization
/// request was processed.
fn type_infer_attribute_expr(
    monomorphizer: &mut Monomorphizer,
    new_func_id: FuncId,
    globals: &BTreeMap<
        noirc_frontend::monomorphization::ast::GlobalId,
        (
            String,
            noirc_frontend::monomorphization::ast::Type,
            noirc_frontend::monomorphization::ast::Expression,
        ),
    >,
    min_available_id: Rc<RefCell<u32>>,
    parameters_hir_types: &HashMap<String, noirc_frontend::Type>,
    expr: AnnExpr<Location>,
    new_ids_to_old_ids: &mut HashMap<FuncId, node_interner::FuncId>,
    to_be_added_ghost_attribute: &mut HashSet<FuncId>,
) -> Result<SpannedTypedExpr, MonomorphizationErrorBundle> {
    // Set a reasonable limit to prevent infinite loops in case of a bug.
    const MAX_RETRIES: u32 = 100;
    // TODO(totel): Check if a monomorphization request was send for the same function twice
    // This will indicate that we have reached an infinite recursion point. (Some would call it a fix point)
    for _ in 0..MAX_RETRIES {
        // The following two variables are defined inside of the `for` loop
        // because of the borrow checker.
        let function = &monomorphizer.finished_functions[&new_func_id];
        let state = State {
            function,
            global_constants: &globals,
            functions: &monomorphizer.finished_functions,
            min_local_id: min_available_id.clone(),
        };

        let expr = convert_struct_access_to_tuple_access(expr.clone(), parameters_hir_types)
            .map_err(|e| match e {
                formal_verification::convert_structs::ResolverOrTypeError::ResolverError(
                    resolver_error,
                ) => MonomorphizationErrorBundle::ResolverError(resolver_error),
                formal_verification::convert_structs::ResolverOrTypeError::TypeError(
                    type_check_error,
                ) => MonomorphizationErrorBundle::TypeError(type_check_error),
            })?;

        match type_infer(&state, expr.clone()) {
            Ok(typed_expr) => {
                // Success, return immediately.
                return Ok(typed_expr);
            }
            Err(type_error) => match type_error {
                TypeInferenceError::MonomorphizationRequest(request) => {
                    // This is a recoverable error. Try to resolve it.
                    monomorphize_one_function(
                        &request.function_identifier,
                        request.param_types,
                        monomorphizer,
                        new_ids_to_old_ids,
                        to_be_added_ghost_attribute,
                    )?;
                    // After monomorphizing the function try to type infer again.
                }
                other_error => {
                    // This is an unrecoverable error, return immediately.
                    return Err(MonomorphizationErrorBundle::from(other_error));
                }
            },
        }
    }

    // If the loop finishes, we've exceeded the retry limit. This indicates a likely bug.
    // TODO(totel): Define a better error
    panic!("Monomorphization limit reached")
}

fn monomorphize_one_function(
    func_name: &str,
    param_types: Vec<OptionalType>,
    monomorphizer: &mut Monomorphizer,
    new_ids_to_old_ids: &mut HashMap<FuncId, node_interner::FuncId>,
    to_be_added_ghost_attribute: &mut HashSet<FuncId>,
) -> Result<(), MonomorphizationErrorBundle> {
    let func_id = monomorphizer.interner.find_function(func_name).expect(&format!(
        "The provided function name {}, was not found during the completion of MonomorphRequest",
        func_name
    ));
    let func_id_as_expr_id = monomorphizer.interner.function(&func_id).as_expr();

    let pseudo_args: Vec<ExprId> = std::iter::repeat_with(|| {
        monomorphizer.interner.push_expr_full(
            HirExpression::Literal(HirLiteral::Bool(true)),
            Location::dummy(),
            noirc_frontend::Type::Bool,
        )
    })
    .take(param_types.len())
    .collect();

    let pseudo_call_expr = HirExpression::Call(HirCallExpression {
        func: func_id_as_expr_id,
        arguments: pseudo_args,
        location: Location::dummy(),
        is_macro_call: false,
    });

    let pseudo_call_expr_id = monomorphizer.interner.push_expr_full(
        pseudo_call_expr,
        Location::dummy(),
        noirc_frontend::Type::Unit,
    );

    let mut typ_bindings = noirc_frontend::Type::Unit.instantiate(&monomorphizer.interner).1;

    // Bind generic types to the type used in the function call
    monomorphizer
        .interner
        .function_meta(&func_id)
        .parameters
        .0
        .iter()
        .map(|(_pattern, typ, _visibility)| typ)
        .enumerate()
        .filter_map(|(pos, typ)| match typ {
            noirc_frontend::Type::NamedGeneric(named_generic) => {
                Some((pos, &named_generic.type_var))
            }
            noirc_frontend::Type::TypeVariable(type_var) => Some((pos, type_var)),
            _ => None,
        })
        .for_each(|(param_index, type_var)| {
            // The last argument of method `.insert` is the important one
            typ_bindings.insert(
                type_var.id(),
                (
                    type_var.clone(),
                    Kind::Normal,
                    convert_mast_to_noir_type(
                        param_types[param_index]
                            .clone()
                            .unwrap_or(noirc_frontend::monomorphization::ast::Type::Field),
                    ),
                ),
            );
        });

    monomorphizer.interner.store_instantiation_bindings(pseudo_call_expr_id, typ_bindings);

    // NOTE: `queue_function` was made public by us
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
            .map_err(MonomorphizationErrorBundle::MonomorphizationError)?;

        monomorphizer
            .function(next_fn_id, new_id, location)
            .map_err(MonomorphizationErrorBundle::MonomorphizationError)?;
        new_ids_to_old_ids.insert(new_id, next_fn_id);
        undo_instantiation_bindings(impl_bindings);
        undo_instantiation_bindings(bindings);

        if has_ghost_attribute(monomorphizer.interner.function_modifiers(&func_id)) {
            to_be_added_ghost_attribute.insert(new_id);
        } else if func_name == OLD {
            // Note: We don't want to monomorphize `fv_std::old` into
            // a ghost function because we may get a verifier error for
            // having a ghost function with a mut reference parameter type.

            // The function call to `fv_std::old` gets converted to a special
            // Verus expressions anyways. Therefore the function `old` is not
            // being actually called anywhere in the code.

            // Therefore we don't want to produce an error
        } else {
            // A non-ghost function has been called in a FV annotation.
            // This isn't allowed and we have to produce an adequate error.

            return Err(MonomorphizationErrorBundle::FvError(
                errors::FvMonomorphizationError::ExecInSpecError {
                    func_name: func_name.to_string(),
                    location: monomorphizer.interner.function_meta(&func_id).location,
                },
            ));
        }
    }

    Ok(())
}

fn has_ghost_attribute(func_modifiers: &FunctionModifiers) -> bool {
    func_modifiers.attributes.secondary.iter().any(|SecondaryAttribute { kind, location: _ }| {
        matches!(kind, SecondaryAttributeKind::Tag(tag) if tag == "ghost")
    })
}

fn collect_parameter_hir_types(
    monomorphizer: &Monomorphizer,
    old_id: &node_interner::FuncId,
) -> HashMap<String, noirc_frontend::Type> {
    let interner = &monomorphizer.interner;
    let func_meta = interner.function_meta(old_id);

    let mut struct_arguments: HashMap<String, noirc_frontend::Type> = func_meta
        .parameters
        .0
        .iter()
        .map(|(hir_pattern, typ, _)| {
            // Extract identifier from the pattern
            let ident = match hir_pattern {
                HirPattern::Identifier(ident) => ident,
                HirPattern::Mutable(inner, _) => match inner.as_ref() {
                    HirPattern::Identifier(ident) => ident,
                    // NOTE: We assume that only the Hir patterns "Identifier" and "Mutable"
                    // appear for function parameters
                    other => unreachable!(
                        "Expected HirPattern::Identifier inside HirPattern::Mutable, got: {:?}",
                        other
                    ),
                },
                other => unreachable!(
                    "Expected HirPattern::Identifier or HirPattern::Mutable, got: {:?}",
                    other
                ),
            };

            (interner.definition(ident.id).name.clone(), typ.clone())
        })
        .collect();

    struct_arguments.insert("result".to_string(), func_meta.return_type().clone());

    struct_arguments
}
