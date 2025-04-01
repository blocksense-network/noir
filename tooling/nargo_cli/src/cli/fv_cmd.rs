use clap::Args;
use nargo::workspace::Workspace;
use nargo_toml::PackageSelection;
use noirc_driver::CompileOptions;

use crate::errors::CliError;

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
    todo!()
}
