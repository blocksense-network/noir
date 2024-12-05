use clap::Args;
use fm::{FileId, FileManager};
use nargo::{
    insert_all_files_for_workspace_into_file_manager, ops::report_errors, parse_all,
    prepare_package,
};
use nargo_toml::{get_package_manifest, resolve_workspace_from_toml, PackageSelection};
use noirc_driver::{
    file_manager_with_stdlib, link_to_debug_crate, CompileOptions, NOIR_ARTIFACT_VERSION_STRING,
};
use noirc_errors::reporter::ReportedErrors;
use noirc_errors::{CustomDiagnostic, DiagnosticKind, FileDiagnostic, Span};
use noirc_frontend::{debug::DebugInstrumenter, graph::CrateName};
use serde::Deserialize;
use std::result::Result::Ok;
use std::{
    io::Write,
    process::{Command, Stdio},
};
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

        z3_verify(noir_program_to_vir, &workspace_file_manager, args.compile_options.deny_warnings)?
    }

    Ok(())
}

/// Verifies the VIR crate which the Noir code was transformed into
pub(crate) fn z3_verify(
    vir_krate: Krate,
    workspace_file_manager: &FileManager,
    deny_warnings: bool,
) -> Result<(), CliError> {
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

    let mut smt_outputs: Vec<SmtOutput> = Vec::new();
    let lines: Vec<String> = stderr_output.lines().map(String::from).collect();
    for line in &lines {
        if let Ok(smt_output) = serde_json::from_str::<SmtOutput>(&line) {
            smt_outputs.push(smt_output);
        } else {
            println!("Failed to deserialize: {}", line);
        }
    }

    let verification_diagnostics: Vec<FileDiagnostic> = smt_outputs
        .into_iter()
        .map(|smt_output| smt_output_to_diagnostic(smt_output, &workspace_file_manager))
        .collect::<Result<_, _>>()?;
    let reported_errors: ReportedErrors = noirc_errors::reporter::report_all(
        workspace_file_manager.as_file_map(),
        &verification_diagnostics,
        deny_warnings,
        false,
    );

    if reported_errors.error_count == 0 {
        println!("Verification successful!");
        Ok(())
    } else {
        Err(CliError::VerificationFail(String::new()))
    }
}

#[derive(Deserialize)]
struct ErrorBlock {
    error_message: String,
    error_span: String,
    secondary_message: String,
}

#[derive(Deserialize)]
struct WarningBlock {
    warning_message: String,
}

#[derive(Deserialize)]
enum SmtOutput {
    Error(ErrorBlock),
    Warning(WarningBlock),
    Note(String),
    AirMessage(String),
}

fn smt_output_to_diagnostic(
    smt_output: SmtOutput,
    workspace_file_manager: &FileManager,
) -> Result<FileDiagnostic, CliError> {
    let default_file_id = workspace_file_manager
        .as_file_map()
        .all_file_ids()
        .last()
        .unwrap_or(&FileId::dummy())
        .clone();

    match smt_output {
        SmtOutput::Error(error_block) => {
            if let Ok((start_byte, final_byte, file_id)) = convert_span(&error_block.error_span) {
                let diagnostic = CustomDiagnostic::simple_error(
                    error_block.error_message,
                    error_block.secondary_message,
                    Span::inclusive(start_byte, final_byte),
                );
                Ok(FileDiagnostic { file_id: FileId::new(file_id), diagnostic })
            } else {
                Ok(FileDiagnostic {
                    file_id: default_file_id,
                    diagnostic: CustomDiagnostic::from_message(&error_block.error_message),
                })
            }
        }
        SmtOutput::Warning(warning_block) => Ok(FileDiagnostic {
            file_id: default_file_id,
            diagnostic: CustomDiagnostic::from_message_kind(
                &warning_block.warning_message,
                DiagnosticKind::Warning,
            ),
        }),
        SmtOutput::Note(message) => Ok(FileDiagnostic {
            file_id: default_file_id,
            diagnostic: CustomDiagnostic::from_message_kind(&message, DiagnosticKind::Info),
        }),
        SmtOutput::AirMessage(message) => Err(CliError::VerificationCrash(message)),
    }
}

fn convert_span(input: &str) -> Result<(u32, u32, usize), Box<dyn std::error::Error>> {
    if input.is_empty() {
        return Err("Span is empty".into());
    }
    let trimmed = input.trim_matches(|c| c == '(' || c == ')');
    let parts: Vec<&str> = trimmed.split(',').map(str::trim).collect();

    if parts.len() != 3 {
        return Err("Span must have exactly three elements".into());
    }

    let start_byte = parts[0].parse::<u32>()?;
    let final_byte = parts[1].parse::<u32>()?;
    let file_id = parts[2].parse::<usize>()?;

    Ok((start_byte, final_byte, file_id))
}
