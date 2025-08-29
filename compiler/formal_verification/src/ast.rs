use std::fmt::{Debug, Display};

use noirc_errors::Location;
use noirc_frontend::monomorphization::ast::Type as NoirType;
use num_bigint::BigInt;

pub type Identifier = String;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ExprF<R> {
    BinaryOp { op: BinaryOp, expr_left: R, expr_right: R },
    UnaryOp { op: UnaryOp, expr: R },
    Parenthesised { expr: R },
    Quantified { quantifier: Quantifier, variables: Vec<Variable>, expr: R },
    FnCall { name: Identifier, args: Vec<R> },
    Index { expr: R, index: R },
    TupleAccess { expr: R, index: u32 },
    StructureAccess { expr: R, field: Identifier },
    Cast { expr: R, target: NoirType },
    Literal { value: Literal },
    Tuple { exprs: Vec<R> },
    Array { exprs: Vec<R> },
    Variable(Variable),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AnnExpr<A> {
    pub ann: A,
    pub expr: Box<ExprF<AnnExpr<A>>>,
}

pub type SpannedTypedExpr = AnnExpr<(Location, NoirType)>;
pub type SpannedExpr = AnnExpr<Location>;
pub type OffsetExpr = AnnExpr<(u32, u32)>;

#[derive(Clone, derive_more::Debug, PartialEq, Eq)]
#[debug("{_0:?}")]
pub struct RawExpr(pub Box<ExprF<RawExpr>>);

pub fn strip_ann<T>(expr: AnnExpr<T>) -> RawExpr {
    cata(expr, &|_, expr| RawExpr(Box::new(expr)))
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Literal {
    Bool(bool),
    Int(BigInt),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Quantifier {
    Forall,
    Exists,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum UnaryOp {
    // neither
    Dereference,

    // Arithmetic and Boolean
    Not,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BinaryOp {
    // pure Arithmetic (data -> data)
    Mul,
    Div,
    Mod,
    Add,
    Sub,
    ShiftLeft,
    ShiftRight,

    // pure Predicates (data -> bool)
    Eq,
    Neq,
    Lt,
    Le,
    Gt,
    Ge,

    // pure Boolean (bool -> bool)
    Implies,

    // Arithmentic and Boolean
    And,
    Or,
    Xor,
}

impl BinaryOp {
    /// An operator is "arithmetic" if it performs an arithmetic operation
    /// (e.g., +, -, *) and returns a value of the same numeric type as its operands.
    pub fn is_arithmetic(&self) -> bool {
        matches!(self, Self::Add | Self::Sub | Self::Mul | Self::Div | Self::Mod)
    }

    /// An operator is "bitwise" if it performs a bitwise operation (e.g., &, |, ^).
    /// In this language, these can operate on both integers and booleans.
    pub fn is_bitwise(&self) -> bool {
        matches!(self, Self::And | Self::Or | Self::Xor)
    }

    /// An operator is a "shift" if it performs a bit shift (e.g., <<, >>).
    /// These have unique type rules, requiring a numeric type on the left
    /// and a `u8` on the right.
    pub fn is_shift(&self) -> bool {
        matches!(self, Self::ShiftLeft | Self::ShiftRight)
    }

    /// An operator is a "comparison" if it compares two values and always returns a boolean.
    pub fn is_comparison(&self) -> bool {
        matches!(self, Self::Eq | Self::Neq | Self::Lt | Self::Le | Self::Gt | Self::Ge)
    }

    /// An operator is "logical" if its operands are expected to be booleans.
    /// This includes `==>` and the bitwise operators when used in a boolean context.
    pub fn is_logical(&self) -> bool {
        matches!(self, Self::Implies | Self::And | Self::Or)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Variable {
    pub path: Vec<Identifier>,
    pub name: Identifier,
    pub id: Option<u32>,
}

/*
* cata stuff
* */

pub fn fmap<A, B>(expr: ExprF<A>, cata_fn: &impl Fn(A) -> B) -> ExprF<B> {
    match expr {
        ExprF::BinaryOp { op, expr_left, expr_right } => {
            ExprF::BinaryOp { op, expr_left: cata_fn(expr_left), expr_right: cata_fn(expr_right) }
        }
        ExprF::UnaryOp { op, expr } => ExprF::UnaryOp { op, expr: cata_fn(expr) },
        ExprF::Parenthesised { expr } => ExprF::Parenthesised { expr: cata_fn(expr) },
        ExprF::Quantified { quantifier, variables, expr } => {
            ExprF::Quantified { quantifier, variables, expr: cata_fn(expr) }
        }
        ExprF::FnCall { name, args } => {
            ExprF::FnCall { name, args: args.into_iter().map(cata_fn).collect() }
        }
        ExprF::Index { expr, index } => ExprF::Index { expr: cata_fn(expr), index: cata_fn(index) },
        ExprF::TupleAccess { expr, index } => ExprF::TupleAccess { expr: cata_fn(expr), index },
        ExprF::StructureAccess { expr, field } => {
            ExprF::StructureAccess { expr: cata_fn(expr), field }
        }
        ExprF::Cast { expr, target } => ExprF::Cast { expr: cata_fn(expr), target },
        ExprF::Literal { value } => ExprF::Literal { value },
        ExprF::Tuple { exprs } => {
            ExprF::Tuple { exprs: exprs.into_iter().map(cata_fn).collect::<Vec<_>>() }
        }
        ExprF::Array { exprs } => {
            ExprF::Array { exprs: exprs.into_iter().map(cata_fn).collect::<Vec<_>>() }
        }
        ExprF::Variable(Variable { path, name, id }) => {
            ExprF::Variable(Variable { path, name, id })
        }
    }
}

fn try_fmap<A, B, E>(expr: ExprF<A>, cata_fn: &impl Fn(A) -> Result<B, E>) -> Result<ExprF<B>, E> {
    Ok(match expr {
        ExprF::BinaryOp { op, expr_left, expr_right } => {
            ExprF::BinaryOp { op, expr_left: cata_fn(expr_left)?, expr_right: cata_fn(expr_right)? }
        }
        ExprF::UnaryOp { op, expr } => ExprF::UnaryOp { op, expr: cata_fn(expr)? },
        ExprF::Parenthesised { expr } => ExprF::Parenthesised { expr: cata_fn(expr)? },
        ExprF::Quantified { quantifier, variables, expr } => {
            ExprF::Quantified { quantifier, variables, expr: cata_fn(expr)? }
        }
        ExprF::FnCall { name, args } => {
            let processed_args = args.into_iter().map(cata_fn).collect::<Result<Vec<_>, _>>()?;
            ExprF::FnCall { name, args: processed_args }
        }
        ExprF::Index { expr, index } => {
            ExprF::Index { expr: cata_fn(expr)?, index: cata_fn(index)? }
        }
        ExprF::TupleAccess { expr, index } => ExprF::TupleAccess { expr: cata_fn(expr)?, index },
        ExprF::StructureAccess { expr, field } => {
            ExprF::StructureAccess { expr: cata_fn(expr)?, field }
        }
        ExprF::Cast { expr, target } => ExprF::Cast { expr: cata_fn(expr)?, target },
        ExprF::Literal { value } => ExprF::Literal { value },
        ExprF::Tuple { exprs } => {
            ExprF::Tuple { exprs: exprs.into_iter().map(cata_fn).collect::<Result<Vec<_>, _>>()? }
        }
        ExprF::Array { exprs } => {
            ExprF::Array { exprs: exprs.into_iter().map(cata_fn).collect::<Result<Vec<_>, _>>()? }
        }
        ExprF::Variable(Variable { path, name, id }) => {
            ExprF::Variable(Variable { path, name, id })
        }
    })
}

fn try_fmap_mut<A, B, E>(expr: ExprF<A>, cata_fn: &mut impl FnMut(A) -> Result<B, E>) -> Result<ExprF<B>, E> {
    Ok(match expr {
        ExprF::BinaryOp { op, expr_left, expr_right } => {
            ExprF::BinaryOp { op, expr_left: cata_fn(expr_left)?, expr_right: cata_fn(expr_right)? }
        }
        ExprF::UnaryOp { op, expr } => ExprF::UnaryOp { op, expr: cata_fn(expr)? },
        ExprF::Parenthesised { expr } => ExprF::Parenthesised { expr: cata_fn(expr)? },
        ExprF::Quantified { quantifier, variables, expr } => {
            ExprF::Quantified { quantifier, variables, expr: cata_fn(expr)? }
        }
        ExprF::FnCall { name, args } => {
            let processed_args = args.into_iter().map(cata_fn).collect::<Result<Vec<_>, _>>()?;
            ExprF::FnCall { name, args: processed_args }
        }
        ExprF::Index { expr, index } => {
            ExprF::Index {  index: cata_fn(index)? , expr: cata_fn(expr)? }
        }
        ExprF::TupleAccess { expr, index } => ExprF::TupleAccess { expr: cata_fn(expr)?, index },
        ExprF::StructureAccess { expr, field } => {
            ExprF::StructureAccess { expr: cata_fn(expr)?, field }
        }
        ExprF::Cast { expr, target } => ExprF::Cast { expr: cata_fn(expr)?, target },
        ExprF::Literal { value } => ExprF::Literal { value },
        ExprF::Tuple { exprs } => {
            ExprF::Tuple { exprs: exprs.into_iter().map(cata_fn).collect::<Result<Vec<_>, _>>()? }
        }
        ExprF::Array { exprs } => {
            ExprF::Array { exprs: exprs.into_iter().map(cata_fn).collect::<Result<Vec<_>, _>>()? }
        }
        ExprF::Variable(Variable { path, name, id }) => {
            ExprF::Variable(Variable { path, name, id })
        }
    })
}

pub fn cata<A, B>(expr: AnnExpr<A>, algebra: &impl Fn(A, ExprF<B>) -> B) -> B {
    let children_results = fmap(*expr.expr, &|child| cata(child, algebra));

    algebra(expr.ann, children_results)
}

pub fn try_cata<A, B, E>(
    expr: AnnExpr<A>,
    algebra: &impl Fn(A, ExprF<B>) -> Result<B, E>,
) -> Result<B, E> {
    let children_results = try_fmap(*expr.expr, &|child| try_cata(child, algebra))?;

    algebra(expr.ann, children_results)
}

pub fn try_cata_mut<A, B, E>(
    expr: AnnExpr<A>,
    algebra: &mut impl FnMut(A, ExprF<B>) -> Result<B, E>,
) -> Result<B, E> {
    let children_results = try_fmap_mut(*expr.expr, &mut |child| try_cata_mut(child, algebra))?;

    algebra(expr.ann, children_results)
}

pub fn try_contextual_cata<A, B, C, E>(
    expr: AnnExpr<A>,
    initial_context: C,
    update_context: &impl Fn(C, &AnnExpr<A>) -> C,
    algebra: &impl Fn(A, C, ExprF<B>) -> Result<B, E>,
) -> Result<B, E>
where
    C: Clone,
{
    fn recurse<A, B, C, E>(
        expr: AnnExpr<A>,
        context: C,
        update_context: &impl Fn(C, &AnnExpr<A>) -> C,
        algebra: &impl Fn(A, C, ExprF<B>) -> Result<B, E>,
    ) -> Result<B, E>
    where
        C: Clone,
    {
        let new_context = update_context(context, &expr);

        let children_results = try_fmap(*expr.expr, &|child_expr| {
            recurse(child_expr, new_context.clone(), update_context, algebra)
        })?;

        algebra(expr.ann, new_context, children_results)
    }

    recurse(expr, initial_context, update_context, algebra)
}

pub fn try_cata_recoverable<A, B, E>(
    expr: AnnExpr<A>,
    algebra: &impl Fn(A, Result<ExprF<B>, E>) -> Result<B, E>,
) -> Result<B, E> {
    let children_results = try_fmap(*expr.expr, &|child| try_cata_recoverable(child, algebra));

    algebra(expr.ann, children_results)
}

impl Display for Quantifier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            match self {
                Quantifier::Forall => "forall",
                Quantifier::Exists => "exists",
            }
        )
    }
}
