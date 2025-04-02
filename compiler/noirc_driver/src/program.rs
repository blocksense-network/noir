use std::collections::BTreeMap;

use acvm::{FieldElement, acir::circuit::Program};
use fm::FileId;

use noirc_errors::debug_info::DebugInfo;
use noirc_evaluator::errors::SsaReport;
use serde::{Deserialize, Serialize};
use std::hash::{Hash, Hasher};
use vir::ast::Krate;

use super::debug::DebugFile;

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct CompiledProgram {
    pub noir_version: String,
    /// Hash of the [`Program`][noirc_frontend::monomorphization::ast::Program] from which this [`CompiledProgram`]
    /// was compiled.
    ///
    /// Used to short-circuit compilation in the case of the source code not changing since the last compilation.
    pub hash: u64,

    #[serde(
        serialize_with = "Program::serialize_program_base64",
        deserialize_with = "Program::deserialize_program_base64"
    )]
    pub program: Program<FieldElement>,
    pub abi: noirc_abi::Abi,
    pub debug: Vec<DebugInfo>,
    pub file_map: BTreeMap<FileId, DebugFile>,
    pub warnings: Vec<SsaReport>,
    /// Names of the functions in the program. These are used for more informative debugging and benchmarking.
    pub names: Vec<String>,
    /// Names of the unconstrained functions in the program.
    pub brillig_names: Vec<String>,
    /// Verus verifier intermediate representation
    pub verus_vir: Option<Krate>,
}

// Implement the Hash manually because Krate doesn't implement the Hash trait
impl Hash for CompiledProgram {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.noir_version.hash(state);
        self.hash.hash(state);
        self.program.hash(state);
        self.abi.hash(state);
        self.debug.hash(state);
        self.file_map.hash(state);
        self.warnings.hash(state);
        self.names.hash(state);
        self.brillig_names.hash(state);
    }
}
