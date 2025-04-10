use std::fmt::Display;

use iter_extended::vecmap;
use noirc_errors::Span;

use crate::token::Token;

use super::{Expression, Ident, Visitor};

#[derive(Debug, PartialEq, Eq, Clone)]

pub enum QuantifierType {
    Forall,
    Exists,
}

impl From<Token> for QuantifierType {
    fn from(value: Token) -> Self {
        match value {
            Token::Keyword(keyword) => match keyword {
                crate::token::Keyword::Exists => QuantifierType::Exists,

                crate::token::Keyword::Forall => QuantifierType::Forall,

                _ => unreachable!("Parser must guarantee correct tokens"),
            },

            _ => unreachable!("Parser must guarantee correct tokens"),
        }
    }
}

#[derive(Debug, PartialEq, Eq, Clone)]

pub struct QuantifierExpression {
    pub quantifier_type: QuantifierType,
    pub indexes: Vec<Ident>,
    pub body: Expression,
}

impl Display for QuantifierExpression {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let args = vecmap(&self.indexes, ToString::to_string);
        write!(f, "{}(|{}|{})", self.quantifier_type, args.join(", "), self.body)
    }
}

impl Display for QuantifierType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            QuantifierType::Forall => write!(f, "forall"),
            QuantifierType::Exists => write!(f, "exists"),
        }
    }
}

impl QuantifierExpression {
    pub fn accept(&self, span: Span, visitor: &mut impl Visitor) {
        if visitor.visit_quantifier_expression(self, span) {
            self.accept_children(visitor);
        }
    }

    pub fn accept_children(&self, visitor: &mut impl Visitor) {
        self.body.accept(visitor);
    }
}
