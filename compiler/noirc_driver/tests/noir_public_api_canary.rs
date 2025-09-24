// Canary test for Noir's public frontend API that external tooling relies on.
//
// The fixture below mirrors the minimal interaction performed by tools such as
// Verno, Nargo plugins, and language servers:
//   1. create an in-memory crate wired up with the stdlib,
//   2. run `check_crate` to drive parsing, name resolution, and type checking,
//   3. monomorphize the `main` function directly via the public Monomorphizer API.
//
// If any of these steps start returning different structures or require new
// setup, this test will fail. That provides the Noir team with a concrete,
// reproducible example of the API break so downstream tooling can adapt without
// combing through the entire Noir history.

use fm::FileManager;
use noirc_driver::{check_crate, file_manager_with_stdlib, prepare_crate, CompileOptions};
use noirc_frontend::{
    graph::CrateId,
    hir::{Context, ParsedFiles, def_map::parse_file},
    monomorphization::{
        ast::{Program, Expression, Type},
        debug_types::DebugTypeTracker,
        Monomorphizer,
        perform_impl_bindings,
        perform_instantiation_bindings,
        undo_instantiation_bindings,
    },
    ast::{BinaryOpKind, IntegerBitSize},
    debug::DebugInstrumenter,
    shared::{Signedness, Visibility},
};
use std::path::Path;

// Collect parsed results for every file in the file manager (including stdlib)
// so that `Context::new` matches the data layout used by production tooling.
fn parse_all_files(file_manager: &FileManager) -> ParsedFiles {
    file_manager
        .as_file_map()
        .all_file_ids()
        .map(|&file_id| (file_id, parse_file(file_manager, file_id)))
        .collect()
}

// Build a Noir context rooted at an in-memory `main.nr` plus the stdlib, just
// like `nargo` would when compiling a user project.
fn setup_compilation(source: &str) -> (Context<'static, 'static>, CrateId) {
    let root = Path::new("");
    let file_name = Path::new("main.nr");

    let mut file_manager = file_manager_with_stdlib(root);
    file_manager
        .add_file_with_source(file_name, source.to_owned())
        .expect("failed to add source to file manager");

    let parsed_files = parse_all_files(&file_manager);
    let mut context = Context::new(file_manager, parsed_files);
    let crate_id = prepare_crate(&mut context, file_name);

    (context, crate_id)
}

#[test]
fn noir_public_api_canary() {
    let source = r#"
        fn main(x: u32, y: u32) -> pub u32 {
            x + y
        }
    "#;

    let (mut context, crate_id) = setup_compilation(source);
    let options = CompileOptions::default();

    // 1. Check the crate for errors
    let check_result = check_crate(&mut context, crate_id, &options);
    assert!(check_result.is_ok(), "check_crate failed with errors: {:?}", check_result.err());

    // 2. Get the main function
    let main_function_id = context.get_main_function(&crate_id).expect("No main function found");

    // 3. Monomorphize the program
    let mut monomorphizer = Monomorphizer::new(
        &mut context.def_interner,
        DebugTypeTracker::build_from_debug_instrumenter(&DebugInstrumenter::default()),
        false,
    );

    let main_func_sig = monomorphizer.compile_main(main_function_id);
    assert!(main_func_sig.is_ok(), "compile_main failed");

    // Drain the monomorphizer work queue exactly as frontend consumers would.
    while !monomorphizer.queue.is_empty() {
        let (next_fn_id, new_id, bindings, trait_method, is_unconstrained, location) =
            monomorphizer.queue.pop_front().unwrap();
        monomorphizer.locals.clear();
        monomorphizer.in_unconstrained_function = is_unconstrained;

        perform_instantiation_bindings(&bindings);
        let interner = &monomorphizer.interner;
        let impl_bindings = perform_impl_bindings(interner, trait_method, next_fn_id, location);
        assert!(impl_bindings.is_ok(), "perform_impl_bindings failed");

        let function_result = monomorphizer.function(next_fn_id, new_id, location);
        assert!(function_result.is_ok(), "monomorphizer.function failed");

        undo_instantiation_bindings(impl_bindings.unwrap());
        undo_instantiation_bindings(bindings);
    }

    let functions = monomorphizer.finished_functions.into_iter().map(|(_, f)| f).collect();
    let func_sigs = Vec::new(); // Simplified for this test
    let globals = monomorphizer.finished_globals;
    let (debug_variables, debug_functions, debug_types) =
        monomorphizer.debug_type_tracker.extract_vars_and_types();

    let program = Program::new(
        functions,
        func_sigs,
        main_func_sig.unwrap(),
        monomorphizer.return_location,
        globals.into_iter().collect(),
        debug_variables,
        debug_functions,
        debug_types,
    );

    // Basic program shape should remain stable for this fixture.
    assert_eq!(program.functions.len(), 1, "expected only the main function in the program");
    assert!(program.function_signatures.is_empty(), "no auxiliary function signatures expected");
    assert!(program.globals.is_empty(), "no globals expected for this program");
    assert!(program.debug_variables.is_empty());
    assert!(program.debug_functions.is_empty());
    assert!(program.debug_types.is_empty());

    let main_fn = &program.functions[0];
    assert_eq!(main_fn.id, Program::main_id(), "monomorphizer should assign FuncId(0) to main");
    assert_eq!(main_fn.name, "main");
    assert_eq!(main_fn.parameters.len(), 2, "main should take two parameters");

    let expected_type = Type::Integer(Signedness::Unsigned, IntegerBitSize::ThirtyTwo);
    let expected_visibility = Visibility::Private;

    let (x_id, x_mut, x_name, x_type, x_visibility) = &main_fn.parameters[0];
    assert_eq!(x_id.0, 0, "first parameter should map to LocalId(0)");
    assert!(!x_mut, "first parameter should be immutable");
    assert_eq!(x_name, "x");
    assert_eq!(x_type, &expected_type);
    assert_eq!(x_visibility, &expected_visibility);

    let (y_id, y_mut, y_name, y_type, y_visibility) = &main_fn.parameters[1];
    assert_eq!(y_id.0, 1, "second parameter should map to LocalId(1)");
    assert!(!y_mut, "second parameter should be immutable");
    assert_eq!(y_name, "y");
    assert_eq!(y_type, &expected_type);
    assert_eq!(y_visibility, &expected_visibility);

    assert_eq!(main_fn.return_type, expected_type.clone());
    assert_eq!(main_fn.return_visibility, Visibility::Public, "pub return type should surface as public");

    let Expression::Block(body) = &main_fn.body else {
        panic!("expected main body to be a block expression");
    };
    assert_eq!(body.len(), 1, "main block should contain a single expression");

    let Expression::Binary(binary) = &body[0] else {
        panic!("expected body to be a binary expression");
    };
    assert_eq!(binary.operator, BinaryOpKind::Add);

    let Expression::Ident(lhs) = binary.lhs.as_ref() else {
        panic!("expected left operand to be an identifier");
    };
    assert_eq!(lhs.name, "x");
    assert_eq!(lhs.typ, expected_type.clone());

    let Expression::Ident(rhs) = binary.rhs.as_ref() else {
        panic!("expected right operand to be an identifier");
    };
    assert_eq!(rhs.name, "y");
    assert_eq!(rhs.typ, expected_type.clone());

    assert_eq!(program.main_function_signature.0.len(), 2);
    let Some(signature_return) = &program.main_function_signature.1 else {
        panic!("expected main signature to return a value");
    };
    assert!(matches!(
        signature_return,
        noirc_frontend::Type::Integer(Signedness::Unsigned, IntegerBitSize::ThirtyTwo)
    ));
    assert!(program.return_location.is_some(), "main should have a recorded return location");
}
