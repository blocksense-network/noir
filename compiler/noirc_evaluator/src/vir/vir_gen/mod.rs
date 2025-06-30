pub mod attribute;
pub mod expr_to_vir;
pub mod function;
pub mod globals;

use function::build_funx;
use noirc_errors::Location;
use noirc_frontend::monomorphization::ast::Program;
use serde::{Deserialize, Serialize};
use std::{fmt::Display, sync::Arc};
use vir::{
    ast::{Krate, KrateX, ModuleX, PathX},
    def::Spanned,
    messages::Span,
};

use crate::vir::vir_gen::{expr_to_vir::expression_location, globals::build_global_const_x};

fn encode_span_to_string(location: Location) -> String {
    let stringified_span: String = format!("{}, {}", location.span.start(), location.span.end());
    let stringified_file_id: String = format!("{}", location.file.as_usize());

    format!("({}, {})", stringified_span, stringified_file_id)
}

fn build_span_no_id(debug_string: String, span: Option<Location>) -> Span {
    build_span(0, debug_string, span)
}

fn build_span(id: u32, debug_string: String, span: Option<Location>) -> Span {
    let encoded_span = span.map(encode_span_to_string).unwrap_or_default();
    Span {
        raw_span: Arc::new(()), // Currently unusable because of hard to resolve bug
        id: id as u64,          // Monomorphized AST doesn't have ids
        data: Vec::new(),
        as_string: encoded_span + &debug_string, // It's used as backup if there is no other way to show where the error comes from.
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BuildingKrateError {
    Error(String),
}

impl Display for BuildingKrateError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BuildingKrateError::Error(msg) => write!(f, "{}", msg),
        }
    }
}

pub fn build_krate(program: Program) -> Result<Krate, BuildingKrateError> {
    let mut vir = KrateX {
        functions: Vec::new(),
        reveal_groups: Vec::new(),
        datatypes: Vec::new(),
        traits: Vec::new(),
        trait_impls: Vec::new(),
        assoc_type_impls: Vec::new(),
        modules: Vec::new(),
        external_fns: Vec::new(),
        external_types: Vec::new(),
        path_as_rust_names: vir::ast_util::get_path_as_rust_names_for_krate(&Arc::new(
            vir::def::VERUSLIB.to_string(),
        )),
        arch: vir::ast::Arch { word_bits: vir::ast::ArchWordBits::Either32Or64 },
    };

    // There are no modules in the Noir's monomorphized AST.
    // Therefore we are creating a default module and all functions will be a part of it.
    let module = Spanned::new(
        build_span_no_id(String::from("module"), None),
        ModuleX {
            path: Arc::new(PathX {
                krate: None,
                segments: Arc::new(vec![Arc::new(String::from("Mon. AST"))]),
            }),
            reveals: None,
        },
    );

    // Insert global constants as functions. This is how Verus processes constants
    vir.functions.extend(program.globals.iter().map(
        |(global_id, (name, ast_type, expression))| {
            let global_const_x =
                build_global_const_x(name, ast_type, expression, &module, &program.globals);
            Spanned::new(
                build_span(
                    global_id.0,
                    format!("Global const {} = {}", name, expression),
                    expression_location(expression),
                ),
                global_const_x,
            )
        },
    ));

    for function in &program.functions {
        let func_x = build_funx(function, &module, &program.globals)?;
        let function = Spanned::new(
            build_span(
                function.id.0,
                format!("Function({}) with name {}", function.id.0, function.name),
                None,
            ),
            func_x,
        );
        vir.functions.push(function);
    }

    vir.modules.push(module);

    Ok(Arc::new(vir))
}
