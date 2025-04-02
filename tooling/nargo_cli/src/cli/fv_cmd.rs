use clap::Args;
use fm::FileManager;
use nargo::{ops::report_errors, prepare_package, workspace::Workspace};
use nargo_toml::PackageSelection;
use noirc_driver::{link_to_debug_crate, CompilationResult, CompileOptions, CompiledProgram};
use noirc_frontend::debug::DebugInstrumenter;

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
    let (workspace_file_manager, parsed_files) = parse_workspace(&workspace);
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

        z3_verify(compiled_program, &workspace_file_manager, args.compile_options.deny_warnings)?
    }

    Ok(())
}

fn z3_verify(
    compiled_program: CompilationResult<CompiledProgram>,
    workspace_file_manager: &FileManager,
    deny_warnings: bool,
) -> Result<(), CliError> {
    todo!()
}
