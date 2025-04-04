use std::fmt::Display;

use noirc_errors::Location;
use crate::ast::Expression;

#[derive(PartialEq, Eq, Debug, Clone)]
pub enum FormalVerificationAttribute {
    Ensures(EnsuresAttribute),
    Requires(RequiresAttribute),
    Ghost,
}

#[derive(PartialEq, Eq, Debug, Clone)]
pub struct EnsuresAttribute {
    pub body: Expression,
    pub location: Location,
}

#[derive(PartialEq, Eq, Debug, Clone)]
pub struct RequiresAttribute {
    pub body: Expression,
    pub location: Location,
}

impl FormalVerificationAttribute {
    pub fn name(&self) -> String {
        match self {
            FormalVerificationAttribute::Ensures(_) => "ensures".to_string(),
            FormalVerificationAttribute::Requires(_) => "requires".to_string(),
            FormalVerificationAttribute::Ghost => "ghost".to_string(),
        }
    }
}

impl Display for FormalVerificationAttribute {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FormalVerificationAttribute::Ensures(ens_attribute) => write!(f, "ensures({})", ens_attribute.body.to_string()),
            FormalVerificationAttribute::Requires(req_attribute) => write!(f, "requires({})", req_attribute.body.to_string()),
            FormalVerificationAttribute::Ghost => write!(f, "ghost"),
        }
    }
}