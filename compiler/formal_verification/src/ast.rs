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

pub type SpannedOptionallyTypedExpr = AnnExpr<(Location, Option<NoirType>)>;
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
    pub fn is_arithmetic(&self) -> bool {
        matches!(
            self,
            // pure
            BinaryOp::Mul
                | BinaryOp::Div
                | BinaryOp::Mod
                | BinaryOp::Add
                | BinaryOp::Sub
                | BinaryOp::ShiftLeft
                | BinaryOp::ShiftRight
            // generic
                | BinaryOp::And
                | BinaryOp::Or
                | BinaryOp::Xor
        )
    }

    pub fn is_predicate(&self) -> bool {
        matches!(
            self,
            BinaryOp::Eq
                | BinaryOp::Neq
                | BinaryOp::Lt
                | BinaryOp::Le
                | BinaryOp::Gt
                | BinaryOp::Ge
        )
    }

    pub fn is_boolean(&self) -> bool {
        matches!(
            self,
            // pure
            BinaryOp::Implies
            // generic
                | BinaryOp::And
                | BinaryOp::Or
                | BinaryOp::Xor
        )
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

pub fn fmap<A, B>(expr: ExprF<A>, cata_fn: &dyn Fn(A) -> B) -> ExprF<B> {
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
        ExprF::Index { expr, index } => {
            ExprF::Index { expr: cata_fn(expr), index: cata_fn(index) }
        }
        ExprF::TupleAccess { expr, index } => ExprF::TupleAccess { expr: cata_fn(expr), index },
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

fn try_fmap<A, B, E>(expr: ExprF<A>, cata_fn: &dyn Fn(A) -> Result<B, E>) -> Result<ExprF<B>, E> {
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
        ExprF::Cast { expr, target } => ExprF::Cast { expr: cata_fn(expr)?, target },
        ExprF::Literal { value } => ExprF::Literal { value },
        ExprF::Tuple { exprs } => {
            ExprF::Tuple { exprs: exprs.into_iter().map(cata_fn).collect::<Result<Vec<_>, _>>()? }
        }
        ExprF::Array { exprs } => ExprF::Array {
            exprs: exprs.into_iter().map(cata_fn).collect::<Result<Vec<_>, _>>()?,
        },
        ExprF::Variable(Variable { path, name, id }) => {
            ExprF::Variable(Variable { path, name, id })
        }
    })
}

// TODO: `impl` vs` `dyn` for `cata_fn`
pub fn cata<A, B>(expr: AnnExpr<A>, algebra: &dyn Fn(A, ExprF<B>) -> B) -> B {
    let children_results = fmap(*expr.expr, &|child| cata(child, algebra));

    algebra(expr.ann, children_results)
}

pub fn try_cata<A, B, E>(
    expr: AnnExpr<A>,
    algebra: &dyn Fn(A, ExprF<B>) -> Result<B, E>,
) -> Result<B, E> {
    let children_results = try_fmap(*expr.expr, &|child| try_cata(child, algebra))?;

    algebra(expr.ann, children_results)
}

pub fn try_contextual_cata<A, B, C, E>(
    expr: AnnExpr<A>,
    initial_context: C,
    update_context: &dyn Fn(C, &AnnExpr<A>) -> C,
    algebra: &dyn Fn(A, C, ExprF<B>) -> Result<B, E>,
) -> Result<B, E>
where
    C: Clone,
{
    fn recurse<A, B, C, E>(
        expr: AnnExpr<A>,
        context: C,
        update_context: &dyn Fn(C, &AnnExpr<A>) -> C,
        algebra: &dyn Fn(A, C, ExprF<B>) -> Result<B, E>,
    ) -> Result<B, E>
    where
        C: Clone,
    {
        let children_context = update_context(context.clone(), &expr);

        let children_results = try_fmap(*expr.expr, &|child_expr| {
            recurse(child_expr, children_context.clone(), update_context, algebra)
        })?;

        algebra(expr.ann, context, children_results)
    }

    recurse(expr, initial_context, update_context, algebra)
}

pub fn try_cata_recoverable<A, B, E>(
    expr: AnnExpr<A>,
    algebra: &dyn Fn(A, Result<ExprF<B>, E>) -> Result<B, E>,
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
