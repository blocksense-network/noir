use std::{
    io::Write,
    process::{Command, Stdio},
};

use clap::Args;
use nargo::{
    insert_all_files_for_workspace_into_file_manager, ops::report_errors, parse_all,
    prepare_package,
};
use nargo_toml::{get_package_manifest, resolve_workspace_from_toml, PackageSelection};
use noirc_driver::{
    file_manager_with_stdlib, link_to_debug_crate, CompileOptions, NOIR_ARTIFACT_VERSION_STRING,
};
use noirc_frontend::{debug::DebugInstrumenter, graph::CrateName};
use vir::ast::Krate;

use crate::errors::CliError;

use super::NargoConfig;

/// Perform formal verification on a program
#[derive(Debug, Clone, Args)]
#[clap(visible_alias = "fv")]
pub(crate) struct FormalVerifyCommand {
    /// The name of the package to formally verify
    #[clap(long, conflicts_with = "workspace")]
    package: Option<CrateName>,

    /// Formally verify all packages in the workspace
    #[clap(long, conflicts_with = "package")]
    workspace: bool,

    // This is necessary for compile functions
    #[clap(flatten)]
    compile_options: CompileOptions,
}

pub(crate) fn run(args: FormalVerifyCommand, config: NargoConfig) -> Result<(), CliError> {
    let toml_path = get_package_manifest(&config.program_dir)?;
    let default_selection =
        if args.workspace { PackageSelection::All } else { PackageSelection::DefaultOrAll };
    let selection = args.package.map_or(default_selection, PackageSelection::Selected);
    let workspace = resolve_workspace_from_toml(
        &toml_path,
        selection,
        Some(NOIR_ARTIFACT_VERSION_STRING.to_string()),
    )?;

    let mut workspace_file_manager = file_manager_with_stdlib(&workspace.root_dir);
    insert_all_files_for_workspace_into_file_manager(&workspace, &mut workspace_file_manager);
    let parsed_files = parse_all(&workspace_file_manager);

    let binary_packages = workspace.into_iter().filter(|package| package.is_binary());
    for package in binary_packages {
        let (mut context, crate_id) =
            prepare_package(&workspace_file_manager, &parsed_files, package);
        link_to_debug_crate(&mut context, crate_id);
        context.debug_instrumenter = DebugInstrumenter::default();
        context.package_build_path = workspace.package_build_path(package);
        context.perform_formal_verification = true;

        let compiled_program =
            noirc_driver::compile_main(&mut context, crate_id, &args.compile_options, None, false);

        report_errors(
            compiled_program.clone(),
            &workspace_file_manager,
            args.compile_options.deny_warnings,
            true, // We don't want to report compile related warnings
        )?;

        let noir_program_to_vir = compiled_program.unwrap().0.verus_vir.unwrap();

        z3_verify(noir_program_to_vir)?
    }

    Ok(())
}

/// Verifies the VIR crate which the Noir code was transformed into
pub(crate) fn z3_verify(vir_krate: Krate) -> Result<(), CliError> {
    let serialized_vir_krate = serde_json::to_string(&vir_krate).expect("Failed to serialize");

    let mut child = Command::new("venir")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("Failed to start the Venir binary. Please run the command nix develop");

    if let Some(mut stdin) = child.stdin.take() {
        stdin.write_all(serialized_vir_krate.as_bytes()).expect("Failed to write to Venir stdin");
    }

    let output = child.wait_with_output().expect("Failed to read Venir stdout");

    let stdout_output = String::from_utf8_lossy(&output.stdout);
    if !stdout_output.is_empty() {
        println!("{}", stdout_output);
    }

    let stderr_output = String::from_utf8_lossy(&output.stderr);

    if !output.status.success() {
        Err(CliError::VerificationCrash(stderr_output.trim().to_string()))?
    }
    if stderr_output.contains("Error:") {
        Err(CliError::VerificationFail(stderr_output.trim().to_string()))?
    }

    println!("Verification successful!");
    Ok(())
}
