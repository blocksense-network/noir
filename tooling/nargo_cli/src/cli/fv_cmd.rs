use std::{
    io::Write,
    process::{Command, Stdio},
};

use clap::Args;
use fm::{FileId, FileManager, FileMap};
use fv_bridge::compile_and_build_vir_krate;
use nargo::{ops::report_errors, prepare_package, workspace::Workspace};
use nargo_toml::PackageSelection;
use noirc_driver::{CompileOptions, link_to_debug_crate};
use noirc_errors::{CustomDiagnostic, DiagnosticKind, Location, Span, reporter::ReportedErrors};
use noirc_frontend::debug::DebugInstrumenter;
use serde::Deserialize;
use vir::ast::Krate;

use crate::{cli::compile_cmd::parse_workspace, errors::CliError};

use super::{LockType, PackageOptions, WorkspaceCommand};

/// Perform formal verification on a program
#[derive(Debug, Clone, Args)]
#[clap(visible_alias = "fv")]
pub(crate) struct FormalVerifyCommand {
    #[clap(flatten)]
    pub(super) package_options: PackageOptions,

    // This is necessary for compiling packages
    #[clap(flatten)]
    compile_options: CompileOptions,

    /// Emit debug information for the intermediate Verus VIR to stdout
    #[arg(long, hide = true)]
    pub show_vir: bool,

    /// Use the new syntax for FV annotations
    #[arg(long)]
    pub new_syntax: bool,

    // Flags which will be propagated to the Venir binary
    #[clap(last = true)]
    venir_flags: Vec<String>,
}

impl WorkspaceCommand for FormalVerifyCommand {
    fn package_selection(&self) -> PackageSelection {
        self.package_options.package_selection()
    }

    fn lock_type(&self) -> LockType {
        LockType::Exclusive
    }
}

pub(crate) fn run(args: FormalVerifyCommand, workspace: Workspace) -> Result<(), CliError> {
    let (workspace_file_manager, parsed_files) = parse_workspace(&workspace, None);
    let binary_packages = workspace.into_iter().filter(|package| package.is_binary());

    for package in binary_packages {
        let (mut context, crate_id) =
            prepare_package(&workspace_file_manager, &parsed_files, package);
        link_to_debug_crate(&mut context, crate_id);
        context.debug_instrumenter = DebugInstrumenter::default();
        context.package_build_path = workspace.package_build_path(package);
        context.perform_formal_verification = true;

        let krate: Krate = if args.new_syntax {
            // Compile with new syntax and report errors
            report_errors(
                compile_and_build_vir_krate(&mut context, crate_id, &args.compile_options),
                &workspace_file_manager,
                args.compile_options.deny_warnings,
                true,
            )?
        } else {
            // Compile with old syntax, report errors, then extract and unwrap VIR
            let compiled = report_errors(
                noirc_driver::compile_main(&mut context, crate_id, &args.compile_options, None),
                &workspace_file_manager,
                args.compile_options.deny_warnings,
                true,
            )?;
        
            let vir_result = compiled.verus_vir
                .ok_or_else(|| CliError::Generic("Failed to generate VIR with no specific error".into()))?;
        
            let krate = vir_result.map_err(|e| CliError::Generic(e.to_string()))?;
        
            krate
        };
        
        // Shared post-processing
        if args.show_vir {
            println!("Generated VIR:");
            println!("{:#?}", krate);
        }
        
        z3_verify(
            krate,
            &workspace_file_manager,
            args.compile_options.deny_warnings,
            &args.venir_flags,
        )?;
    }

    Ok(())
}

/// Runs the Venir binary and passes the compiled program in VIR format to it
/// Reports all errors produced during Venir (SMT solver) verification
fn z3_verify(
    krate: Krate,
    workspace_file_manager: &FileManager,
    deny_warnings: bool,
    venir_args: &Vec<String>,
) -> Result<(), CliError> {
    let serialized_vir_krate = serde_json::to_string(&krate).expect("Failed to serialize");

    // Run the Venir binary which is used for verifying the vir_krate input.
    let mut child = Command::new("venir")
        .args(venir_args.iter())
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| CliError::Generic(
            format!("Failed to start the Venir binary with the following error message\n{}\nTo fix this issue you can run the command nix develop", e.to_string())
        ))?;

    if let Some(mut stdin) = child.stdin.take() {
        stdin.write_all(serialized_vir_krate.as_bytes()).map_err(|e| {
            CliError::Generic(format!(
                "Failed to write to Venir stdin with the following error message\n{}",
                e.to_string()
            ))
        })?;
    }

    let output = child.wait_with_output().map_err(|e| {
        CliError::Generic(format!("Failed to read Venir stdout\n{}", e.to_string()))
    })?;

    let stdout_output = String::from_utf8_lossy(&output.stdout);
    if !stdout_output.is_empty() {
        println!("{}", stdout_output);
    }

    let stderr_output = String::from_utf8_lossy(&output.stderr);

    let has_crashed = !output.status.success();

    let mut smt_outputs: Vec<SmtOutput> = Vec::new();
    let mut failed_deserialization_lines: Vec<&str> = Vec::new();

    for line in stderr_output.lines() {
        match serde_json::from_str::<SmtOutput>(line) {
            Ok(smt_output) => smt_outputs.push(smt_output),
            Err(_) => failed_deserialization_lines.push(line),
        }
    }

    if !failed_deserialization_lines.is_empty() {
        println!(
            "Failed to deserialize the following lines:\n{}",
            failed_deserialization_lines.join("\n")
        );
        return Err(CliError::Generic(format!(
            "Failed to deserialize all lines outputted by Venir"
        )));
    }
    // Verus reports Notes in reverse order.
    smt_outputs.reverse();

    let mut verification_diagnostics: Vec<CustomDiagnostic> = smt_outputs
        .into_iter()
        .map(|smt_output| smt_output_to_diagnostic(smt_output, &workspace_file_manager))
        .collect();

    // Sort errors by span.
    verification_diagnostics
        .sort_by_key(|diag| diag.secondaries.first().map(|label| label.location.span.start()));

    // Report errors from the verification process.
    let reported_errors: ReportedErrors = noirc_errors::reporter::report_all(
        workspace_file_manager.as_file_map(),
        &verification_diagnostics,
        deny_warnings,
        false,
    );

    if has_crashed {
        return Err(CliError::Generic(format!("Verification crashed!")));
    }

    if reported_errors.error_count > 0 {
        return Err(CliError::Generic(format!(
            "Verification failed due to {} previous errors!",
            reported_errors.error_count,
        )));
    }

    println!("Verification successful!");
    Ok(())
}

/// Part of the Venir output standard.
#[derive(Deserialize)]
struct ErrorBlock {
    error_message: String,
    error_span: String,
    secondary_message: String,
}

/// Part of the Venir output standard.
#[derive(Deserialize)]
struct WarningBlock {
    warning_message: String,
}

/// Part of the Venir output standard.
#[derive(Deserialize)]
struct CrashBlock {
    crash_message: String,
    crash_span: String,
}

/// The possible outputs of the Venir binary.
#[derive(Deserialize)]
enum SmtOutput {
    Error(ErrorBlock),
    Warning(WarningBlock),
    Note(String),
    AirMessage(CrashBlock),
}

/// Maps a Venir output to a Noir diagnostic type error.
fn smt_output_to_diagnostic(
    smt_output: SmtOutput,
    workspace_file_manager: &FileManager,
) -> CustomDiagnostic {
    let default_file_id = workspace_file_manager
        .as_file_map()
        .all_file_ids()
        .last()
        .cloned()
        .unwrap_or(FileId::dummy());

    match smt_output {
        SmtOutput::Error(error_block) => {
            let span = convert_span(&error_block.error_span);
            match span {
                Some((start_byte, final_byte, file_id)) => {
                    let file_id =
                        get_file_id_via_usize(workspace_file_manager.as_file_map(), file_id)
                            .unwrap_or(FileId::dummy());
                    CustomDiagnostic::simple_error(
                        error_block.error_message,
                        error_block.secondary_message,
                        Location::new(Span::inclusive(start_byte, final_byte), file_id),
                    )
                }
                None => CustomDiagnostic::from_message(&error_block.error_message, default_file_id),
            }
        }

        SmtOutput::Warning(warning_block) => CustomDiagnostic {
            file: default_file_id,
            message: warning_block.warning_message,
            secondaries: Vec::new(),
            notes: Vec::new(),
            kind: DiagnosticKind::Warning,
            deprecated: false,
            unnecessary: false,
            call_stack: Default::default(),
        },

        SmtOutput::Note(message) => CustomDiagnostic {
            file: default_file_id,
            message,
            secondaries: Vec::new(),
            notes: Vec::new(),
            kind: DiagnosticKind::Info,
            deprecated: false,
            unnecessary: false,
            call_stack: Default::default(),
        },

        SmtOutput::AirMessage(crash_block) => {
            let span = convert_span(&crash_block.crash_span);
            match span {
                Some((start_byte, final_byte, file_id)) => {
                    let file_id =
                        get_file_id_via_usize(workspace_file_manager.as_file_map(), file_id)
                            .unwrap_or(default_file_id);
                    CustomDiagnostic::simple_error(
                        String::from("Verification crashed"),
                        crash_block.crash_message,
                        Location::new(Span::inclusive(start_byte, final_byte), file_id),
                    )
                }
                None => {
                    // No valid span for crash message; fallback to basic message
                    CustomDiagnostic::from_message(&crash_block.crash_message, default_file_id)
                }
            }
        }
    }
}

/// Returns `FileId` for given id. `FileId` doesn't have a public constructor.
/// Therefore when we get the file id from the Venir error span we have to search
/// the file map and return the matching `FileId`.
fn get_file_id_via_usize(file_map: &FileMap, file_id_as_usize: usize) -> Option<FileId> {
    file_map.all_file_ids().find(|file_id| file_id.as_usize() == file_id_as_usize).cloned()
}

fn convert_span(input: &str) -> Option<(u32, u32, usize)> {
    if input.is_empty() {
        // Input is empty, cannot decode a span
        return None;
    }

    let trimmed = input.trim_matches(|c| c == '(' || c == ')');
    let parts: Vec<&str> = trimmed.split(',').map(str::trim).collect();

    if parts.len() != 3 {
        // Span must have exactly three components: start, end, file_id
        return None;
    }

    let start_byte = parts[0].parse::<u32>().ok()?;
    let final_byte = parts[1].parse::<u32>().ok()?;
    let file_id = parts[2].parse::<usize>().ok()?;

    Some((start_byte, final_byte, file_id))
}
