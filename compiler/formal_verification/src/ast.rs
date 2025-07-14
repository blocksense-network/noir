use noirc_errors::Location;
use noirc_frontend::monomorphization::ast::Type as NoirType;
use num_bigint::BigInt;
use serde::{Deserialize, Serialize};

pub type Identifier = String;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum ExprF<R> {
    Literal { value: Literal },
    Variable { name: Identifier },
    FnCall { name: Identifier, args: Vec<R> },
    Parenthesised { expr: R },
    UnaryOp { op: UnaryOp, expr: R },
    BinaryOp { op: BinaryOp, expr_left: R, expr_right: R },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AnnExpr<A> {
    pub ann: A,
    pub expr: Box<ExprF<AnnExpr<A>>>,
}

pub type SpannedOptionallyTypedExpr = AnnExpr<(Location, Option<NoirType>)>;
pub type SpannedTypedExpr = AnnExpr<(Location, NoirType)>;
pub type SpannedExpr = AnnExpr<Location>;
pub type OffsetExpr = AnnExpr<(u32, u32)>;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum Literal {
    Bool(bool),
    Int(BigInt),
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum UnaryOp {
    Not,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum ArithmeticOp {
    Mul,
    Div,
    Mod,
    Add,
    Sub,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum PredicateOp {
    Eq,
    Neq,
    Lt,
    Le,
    Gt,
    Ge,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum BooleanOp {
    And,
    Or,
    Implies,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum BinaryOp {
    // Arithmetic (data -> data)
    ArithmeticOp(ArithmeticOp),
    // Predicates (data -> bool)
    PredicateOp(PredicateOp),
    // Boolean (bool -> bool)
    BooleanOp(BooleanOp),
}

pub fn fmap<A, B>(expr: ExprF<A>, cata_fn: &dyn Fn(A) -> B) -> ExprF<B> {
    match expr {
        ExprF::Literal { value } => ExprF::Literal { value },
        ExprF::Variable { name } => ExprF::Variable { name },
        ExprF::FnCall { name, args } => {
            let processed_args = args.into_iter().map(cata_fn).collect();
            ExprF::FnCall { name, args: processed_args }
        }
        ExprF::Parenthesised { expr } => ExprF::Parenthesised { expr: cata_fn(expr) },
        ExprF::UnaryOp { op, expr } => ExprF::UnaryOp { op, expr: cata_fn(expr) },
        ExprF::BinaryOp { op, expr_left, expr_right } => {
            ExprF::BinaryOp { op, expr_left: cata_fn(expr_left), expr_right: cata_fn(expr_right) }
        }
    }
}

fn try_fmap<A, B, E>(expr: ExprF<A>, cata_fn: &dyn Fn(A) -> Result<B, E>) -> Result<ExprF<B>, E> {
    Ok(match expr {
        ExprF::Literal { value } => ExprF::Literal { value },
        ExprF::Variable { name } => ExprF::Variable { name },
        ExprF::FnCall { name, args } => {
            let processed_args = args.into_iter().map(cata_fn).collect::<Result<Vec<_>, _>>()?;
            ExprF::FnCall { name, args: processed_args }
        }
        ExprF::Parenthesised { expr } => ExprF::Parenthesised { expr: cata_fn(expr)? },
        ExprF::UnaryOp { op, expr } => ExprF::UnaryOp { op, expr: cata_fn(expr)? },
        ExprF::BinaryOp { op, expr_left, expr_right } => {
            ExprF::BinaryOp { op, expr_left: cata_fn(expr_left)?, expr_right: cata_fn(expr_right)? }
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

pub fn try_cata_recoverable<A, B, E>(
    expr: AnnExpr<A>,
    algebra: &dyn Fn(A, Result<ExprF<B>, E>) -> Result<B, E>,
) -> Result<B, E> {
    let children_results = try_fmap(*expr.expr, &|child| try_cata_recoverable(child, algebra));

    algebra(expr.ann, children_results)
}
