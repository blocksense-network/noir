use std::{
    io::Write,
    process::{Command, Stdio},
};

use clap::Args;
use fm::{FileId, FileManager, FileMap};
use nargo::{ops::report_errors, prepare_package, workspace::Workspace};
use nargo_toml::PackageSelection;
use noirc_driver::{CompileOptions, CompiledProgram, link_to_debug_crate};
use noirc_errors::{CustomDiagnostic, DiagnosticKind, Location, Span, reporter::ReportedErrors};
use noirc_frontend::debug::DebugInstrumenter;
use serde::Deserialize;

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

        let compiled_program =
            noirc_driver::compile_main(&mut context, crate_id, &args.compile_options, None);

        // We want to formally verify only compilable programs
        report_errors(
            compiled_program.clone(),
            &workspace_file_manager,
            args.compile_options.deny_warnings,
            true, // We don't want to report compile related warnings
        )?;

        compiled_program
            .ok()
            .map(|(compiled_program, _)| {
                z3_verify(
                    compiled_program,
                    &workspace_file_manager,
                    args.compile_options.deny_warnings,
                    &args.venir_flags,
                )
            })
            .transpose()?;
    }

    Ok(())
}

/// Runs the Venir binary and passes the compiled program in VIR format to it
/// Reports all errors produced during Venir (SMT solver) verification
fn z3_verify(
    compiled_program: CompiledProgram,
    workspace_file_manager: &FileManager,
    deny_warnings: bool,
    venir_args: &Vec<String>,
) -> Result<(), CliError> {
    let krate = compiled_program
        .verus_vir
        .ok_or_else(|| {
            CliError::Generic(String::from("Failed to generate VIR with no specific error"))
        })?
        .map_err(|e| CliError::Generic(e.to_string()))?;

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
    let lines: Vec<String> = stderr_output.lines().map(String::from).collect();
    let mut failed_deserialization_lines: Vec<&str> = Vec::new();
    for line in &lines {
        if let Ok(smt_output) = serde_json::from_str::<SmtOutput>(&line) {
            smt_outputs.push(smt_output);
        } else {
            failed_deserialization_lines.push(line);
        }
    }
    if !failed_deserialization_lines.is_empty() {
        println!(
            "Failed to deserialize the following lines:\n{}",
            failed_deserialization_lines.join("\n")
        );
        return Err(CliError::Generic(format!("Failed to deserialize all lines outputted by Venir")));
    }

    smt_outputs.reverse();

    let mut verification_diagnostics: Vec<CustomDiagnostic> = smt_outputs
        .into_iter()
        .map(|smt_output| smt_output_to_diagnostic(smt_output, &workspace_file_manager))
        .collect::<Result<_, _>>()?;

    // Sort errors by span.
    verification_diagnostics.sort_by(|a, b| {
        match (a.secondaries.first(), b.secondaries.first()) {
            (None, None) => std::cmp::Ordering::Equal, // Errors with no span are put at the start.
            (None, Some(_)) => std::cmp::Ordering::Less,
            (Some(_), None) => std::cmp::Ordering::Greater,
            (Some(a_custom_label), Some(b_custom_label)) => {
                a_custom_label.location.span.start().cmp(&b_custom_label.location.span.start())
            }
        }
    });

    // Report errors from the verification process.
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
        if has_crashed {
            Err(CliError::Generic(format!("Verification crashed!")))
        } else {
            Err(CliError::Generic(format!(
                "Verification failed due to {} previous errors!",
                reported_errors.error_count
            )))
        }
    }
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
) -> Result<CustomDiagnostic, CliError> {
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
                    Location::new(
                        Span::inclusive(start_byte, final_byte),
                        get_file_id_via_usize(workspace_file_manager.as_file_map(), file_id)
                            .unwrap_or(FileId::dummy()),
                    ),
                );
                Ok(diagnostic)
            } else {
                Ok(CustomDiagnostic::from_message(&error_block.error_message, default_file_id))
            }
        }
        SmtOutput::Warning(warning_block) => Ok(CustomDiagnostic {
            file: default_file_id,
            message: warning_block.warning_message,
            secondaries: Vec::new(),
            notes: Vec::new(),
            kind: DiagnosticKind::Warning,
            deprecated: false,
            unnecessary: false,
            call_stack: Default::default(),
        }),
        SmtOutput::Note(message) => Ok(CustomDiagnostic {
            file: default_file_id,
            message,
            secondaries: Vec::new(),
            notes: Vec::new(),
            kind: DiagnosticKind::Info,
            deprecated: false,
            unnecessary: false,
            call_stack: Default::default(),
        }),
        SmtOutput::AirMessage(crash_block) => {
            let error_span = convert_span(&crash_block.crash_span);

            match error_span {
                Ok((start_byte, final_byte, file_id)) => Ok(CustomDiagnostic::simple_error(
                    String::from("Verification crashed"),
                    crash_block.crash_message,
                    Location::new(
                        Span::inclusive(start_byte, final_byte),
                        get_file_id_via_usize(workspace_file_manager.as_file_map(), file_id)
                            .unwrap_or(default_file_id),
                    ),
                )),
                Err(_) => {
                    // The error means that we have no span for the crash message
                    // Therefore we can ignore it
                    Ok(CustomDiagnostic::from_message(&crash_block.crash_message, default_file_id))
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

// We have encoded the Noir expression span into a string which is attached to
// the relevant VIR expression. We know that this is a bad pattern but all other
// approaches that we tried resulted in failure. Therefore the following function
// decodes span from string.
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
