//! Abstract syntax tree for Jetro v2 expressions.
//!
//! The AST is the contract between the parser and every other v2 layer —
//! the compiler lowers it to opcodes, analyses inspect it for ident-use
//! / purity, and tests build it directly.  Because every component
//! observes `Expr`, its variants are kept deliberately orthogonal:
//! each is a language concept, not an implementation shortcut.
//!
//! `Arc<str>` is used for every identifier so that cloning a name into
//! an opcode is a refcount bump rather than a byte copy.  Recursive
//! sub-expressions are `Box<Expr>` — owned, not shared — since the
//! compiler rewrites the AST in-place (see `reorder_and_operands`).

use std::sync::Arc;

// ── AST nodes ─────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub enum Expr {
    // Literals
    Null,
    Bool(bool),
    Int(i64),
    Float(f64),
    Str(String),
    FString(Vec<FStringPart>),

    // References
    Root,           // $
    Current,        // @
    Ident(String),  // variable or implicit current-item field

    // Navigation chain: base followed by postfix steps
    Chain(Box<Expr>, Vec<Step>),

    // Binary operations
    BinOp(Box<Expr>, BinOp, Box<Expr>),
    UnaryNeg(Box<Expr>),
    Not(Box<Expr>),

    // Kind check: expr kind [not] type
    Kind {
        expr:   Box<Expr>,
        ty:     KindType,
        negate: bool,
    },

    // Null-coalesce: lhs ?| rhs
    Coalesce(Box<Expr>, Box<Expr>),

    // Object / array construction
    Object(Vec<ObjField>),
    Array(Vec<ArrayElem>),

    // Pipeline: base | step1 | step2  or  base -> name | ...
    Pipeline {
        base:  Box<Expr>,
        steps: Vec<PipeStep>,
    },

    // Comprehensions
    ListComp {
        expr: Box<Expr>,
        vars: Vec<String>,
        iter: Box<Expr>,
        cond: Option<Box<Expr>>,
    },
    DictComp {
        key:  Box<Expr>,
        val:  Box<Expr>,
        vars: Vec<String>,
        iter: Box<Expr>,
        cond: Option<Box<Expr>>,
    },
    SetComp {
        expr: Box<Expr>,
        vars: Vec<String>,
        iter: Box<Expr>,
        cond: Option<Box<Expr>>,
    },
    GenComp {
        expr: Box<Expr>,
        vars: Vec<String>,
        iter: Box<Expr>,
        cond: Option<Box<Expr>>,
    },

    // Lambda: lambda x, y: body
    Lambda {
        params: Vec<String>,
        body:   Box<Expr>,
    },

    // Let binding: let x = init in body
    Let {
        name: String,
        init: Box<Expr>,
        body: Box<Expr>,
    },

    // Global function calls
    GlobalCall {
        name: String,
        args: Vec<Arg>,
    },
}

// ── Pipeline step ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub enum PipeStep {
    /// `| expr` — forward current value as new context
    Forward(Expr),
    /// `-> target` — label current value, pass through unchanged
    Bind(BindTarget),
}

#[derive(Debug, Clone)]
pub enum BindTarget {
    Name(String),
    Obj { fields: Vec<String>, rest: Option<String> },
    Arr(Vec<String>),
}

// ── F-string parts ────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub enum FStringPart {
    Lit(String),
    Interp { expr: Expr, fmt: Option<FmtSpec> },
}

#[derive(Debug, Clone)]
pub enum FmtSpec {
    Spec(String),   // :.2f  :>10  etc.
    Pipe(String),   // |upper  |trim  etc.
}

// ── Array element ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub enum ArrayElem {
    Expr(Expr),
    Spread(Expr),
}

// ── Postfix steps ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantifierKind {
    /// `?` — first element or null
    First,
    /// `!` — exactly one element (error if 0 or >1)
    One,
}

#[derive(Debug, Clone)]
pub enum Step {
    Field(String),                     // .field
    OptField(String),                  // ?.field
    Descendant(String),                // ..field
    DescendAll,                        // ..  (all descendants)
    Index(i64),                        // [n]
    DynIndex(Box<Expr>),               // [expr]
    Slice(Option<i64>, Option<i64>),   // [n:m]
    Method(String, Vec<Arg>),          // .method(args)
    OptMethod(String, Vec<Arg>),       // ?.method(args)
    InlineFilter(Box<Expr>),           // {pred}
    Quantifier(QuantifierKind),        // ? or !
}

// ── Function arguments ────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub enum Arg {
    Pos(Expr),
    Named(String, Expr),
}

// ── Object field ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub enum ObjField {
    /// `key: expr`
    Kv { key: String, val: Expr, optional: bool },
    /// `key` — shorthand for `key: key`
    Short(String),
    /// `[expr]: expr` — dynamic key
    Dynamic { key: Expr, val: Expr },
    /// `...expr` — spread object
    Spread(Expr),
}

// ── Binary operators ──────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinOp {
    Add, Sub, Mul, Div, Mod,
    Eq, Neq, Lt, Lte, Gt, Gte,
    Fuzzy,
    And, Or,
}

// ── Kind types ────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KindType {
    Null, Bool, Number, Str, Array, Object,
}

// ── Sort key ──────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct SortKey {
    pub expr: Expr,
    pub desc: bool,
}

impl Expr {
    /// Wrap in a Chain if steps is non-empty, otherwise return self.
    pub fn maybe_chain(self, steps: Vec<Step>) -> Self {
        if steps.is_empty() { self } else { Expr::Chain(Box::new(self), steps) }
    }
}

/// Shared reference to an expression (for lambdas stored in closures).
pub type ExprRef = Arc<Expr>;
