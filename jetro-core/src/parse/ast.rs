//! Abstract syntax tree produced by `parser` and consumed by every
//! downstream layer — compiler, planner, and analyses.
//!
//! Each variant is a language concept, not an implementation shortcut.
//! Identifiers use `Arc<str>` so that cloning a name into an opcode is a
//! refcount bump. Sub-expressions are `Box<Expr>` so the compiler can
//! rewrite them in place (`reorder_and_operands`).

/// Complete expression AST. The parser produces one of these for every
/// syntactically valid Jetro expression.
#[derive(Debug, Clone)]
pub enum Expr {
    /// The `null` literal; evaluates to `Val::Null`.
    Null,
    /// A boolean literal (`true` / `false`).
    Bool(bool),
    /// A 64-bit signed integer literal.
    Int(i64),
    /// A 64-bit floating-point literal.
    Float(f64),
    /// A plain string literal with no interpolation.
    Str(String),
    /// A format string whose parts may contain interpolated sub-expressions.
    FString(Vec<FStringPart>),

    /// The root document binding `$`; evaluates to `Env::root`.
    Root,
    /// The current item binding `@`; evaluates to `Env::current`.
    Current,
    /// A named variable reference resolved from `Env::vars`.
    Ident(String),

    /// A navigation chain: evaluate the base, then apply each `Step` in sequence.
    Chain(Box<Expr>, Vec<Step>),

    /// A binary infix operation; operands may be reordered by the compiler for `And`/`Or`.
    BinOp(Box<Expr>, BinOp, Box<Expr>),
    /// Arithmetic negation of a numeric expression.
    UnaryNeg(Box<Expr>),
    /// Logical negation; coerces the inner value to bool.
    Not(Box<Expr>),

    /// Runtime type-check: `expr is <kind>` or `expr is not <kind>`.
    Kind {
        /// Expression whose runtime type is being checked.
        expr: Box<Expr>,
        /// The target kind to test against.
        ty: KindType,
        /// When `true` the result is inverted (`is not`).
        negate: bool,
    },

    /// Null-coalescing: evaluates left; if null, evaluates and returns right.
    Coalesce(Box<Expr>, Box<Expr>),

    /// Object literal `{ k: v, … }` with optional dynamic keys, spreads, and conditions.
    Object(Vec<ObjField>),
    /// Array literal `[e, …]` where elements may be spread (`...x`).
    Array(Vec<ArrayElem>),

    /// Pipeline expression `base | step1 | step2 | …`; value threads left-to-right.
    Pipeline {
        /// Seed value for the pipeline.
        base: Box<Expr>,
        /// Ordered sequence of forward-pass or bind steps.
        steps: Vec<PipeStep>,
    },

    /// List comprehension `[expr for vars in iter if cond]`.
    ListComp {
        /// Body expression evaluated for each element.
        expr: Box<Expr>,
        /// Binding names introduced by the `for` clause.
        vars: Vec<String>,
        /// Iterator expression supplying elements.
        iter: Box<Expr>,
        /// Optional guard; elements where this is falsy are skipped.
        cond: Option<Box<Expr>>,
    },
    /// Dict comprehension `{key: val for vars in iter if cond}`.
    DictComp {
        /// Expression computing each output key.
        key: Box<Expr>,
        /// Expression computing each output value.
        val: Box<Expr>,
        /// Binding names introduced by the `for` clause.
        vars: Vec<String>,
        /// Iterator expression supplying elements.
        iter: Box<Expr>,
        /// Optional guard; pairs where this is falsy are skipped.
        cond: Option<Box<Expr>>,
    },
    /// Set comprehension `{expr for vars in iter if cond}`; produces a deduplicated array.
    SetComp {
        /// Body expression evaluated for each element.
        expr: Box<Expr>,
        /// Binding names introduced by the `for` clause.
        vars: Vec<String>,
        /// Iterator expression supplying elements.
        iter: Box<Expr>,
        /// Optional guard; elements where this is falsy are skipped.
        cond: Option<Box<Expr>>,
    },
    /// Generator comprehension; currently evaluates lazily into an array like `ListComp`.
    GenComp {
        /// Body expression evaluated for each element.
        expr: Box<Expr>,
        /// Binding names introduced by the `for` clause.
        vars: Vec<String>,
        /// Iterator expression supplying elements.
        iter: Box<Expr>,
        /// Optional guard; elements where this is falsy are skipped.
        cond: Option<Box<Expr>>,
    },

    /// Anonymous function `(p1, p2) -> body`; closed over the current `Env`.
    Lambda {
        /// Ordered parameter names bound when the lambda is called.
        params: Vec<String>,
        /// Expression evaluated in the extended environment.
        body: Box<Expr>,
    },

    /// `let name = init; body` — lexically scoped binding, not a mutation.
    Let {
        /// Name of the new binding.
        name: String,
        /// Initialiser evaluated in the outer scope.
        init: Box<Expr>,
        /// Body evaluated with `name` in scope.
        body: Box<Expr>,
    },

    /// Conditional expression `if cond then then_ else else_`.
    IfElse {
        /// Boolean guard expression.
        cond: Box<Expr>,
        /// Branch taken when `cond` is truthy.
        then_: Box<Expr>,
        /// Branch taken when `cond` is falsy.
        else_: Box<Expr>,
    },

    /// Error-catching expression; evaluates `body`, returns `default` on any error.
    Try {
        /// Expression that may fail at runtime.
        body: Box<Expr>,
        /// Fallback value returned when `body` errors.
        default: Box<Expr>,
    },

    /// Top-level function call `name(args…)` dispatched through the global registry.
    GlobalCall {
        /// Name of the global function to invoke.
        name: String,
        /// Positional and named arguments.
        args: Vec<Arg>,
    },

    /// Explicit type-cast `expr as <type>`; may return null on failure.
    Cast {
        /// Value to cast.
        expr: Box<Expr>,
        /// Target type.
        ty: CastType,
    },

    /// Structural patch `patch root { path: val, … }`; chain-write terminal desugars here.
    Patch {
        /// Document to patch; usually `Expr::Root`.
        root: Box<Expr>,
        /// Ordered list of path/value operations to apply.
        ops: Vec<PatchOp>,
    },

    /// Sentinel emitted by the parser for `.delete()` / `.unset()` terminals.
    /// Reaching the evaluator is a hard error; the compiler must consume it during patch lowering.
    DeleteMark,

    /// Pattern-match expression `match scrutinee { pat when guard -> body, ... }`.
    /// Arms are tested top to bottom; first match wins.
    Match {
        /// Value being matched against the arm patterns.
        scrutinee: Box<Expr>,
        /// Ordered list of `pat -> body` arms with optional guards.
        arms: Vec<MatchArm>,
    },
}

/// One arm of a `Match` expression: a pattern, optional guard, and body.
#[derive(Debug, Clone)]
pub struct MatchArm {
    /// Pattern that the scrutinee must satisfy.
    pub pat: Pat,
    /// Optional `when <expr>` guard evaluated with arm bindings in scope.
    pub guard: Option<Expr>,
    /// Body expression evaluated when this arm fires.
    pub body: Expr,
}

/// Pattern node used in `Match` arms. Patterns describe the shape of a value
/// and may bind subterms into the arm's scope.
#[derive(Debug, Clone)]
pub enum Pat {
    /// Wildcard `_` — matches any value, no binding.
    Wild,
    /// Literal pattern — matches by structural equality with `Lit`.
    Lit(PatLit),
    /// Identifier binding `name` — captures the whole value into `name`.
    Bind(String),
    /// Or-pattern `a | b | c` — matches if any sub-pattern matches.
    Or(Vec<Pat>),
    /// Object pattern `{k: pat, ...}` — every listed key must match. The
    /// runtime always permits extra keys; the `open` flag is currently
    /// informational and left in the AST so future passes can opt into
    /// strict closed-object matching without a grammar change.
    Obj {
        /// Listed key/sub-pattern pairs that must all match.
        fields: Vec<(String, Pat)>,
        /// `true` when the source spelled the trailing `...` rest marker.
        #[allow(dead_code)]
        open: bool,
    },
    /// Array pattern `[a, b, ...rest]` — fixed prefix with optional rest binding.
    Arr { elems: Vec<Pat>, rest: Option<Option<String>> },
    /// Type-kind pattern `name: kind` (e.g. `s: str`) — matches a kind, binds the value.
    Kind { name: Option<String>, kind: KindType },
}

/// Literal sub-form of `Pat::Lit`. Restricted to scalar literals; arbitrary
/// expressions are not allowed in pattern position.
#[derive(Debug, Clone)]
pub enum PatLit {
    /// `null` literal.
    Null,
    /// Boolean literal.
    Bool(bool),
    /// Integer literal.
    Int(i64),
    /// Float literal.
    Float(f64),
    /// String literal.
    Str(String),
}


/// A single write operation inside a `Patch` expression.
#[derive(Debug, Clone)]
pub struct PatchOp {
    /// Navigation path identifying the target node.
    pub path: Vec<PathStep>,
    /// Value to write; `Expr::DeleteMark` removes the node instead.
    pub val: Expr,
    /// Optional guard; the op is skipped when this evaluates to falsy.
    pub cond: Option<Expr>,
}

/// One segment of a patch path — mirrors `Step` but restricted to write-safe forms.
#[derive(Debug, Clone)]
pub enum PathStep {
    /// Static field name lookup.
    Field(String),
    /// Integer index into an array.
    Index(i64),
    /// Runtime-computed index evaluated against the current value.
    DynIndex(Expr),
    /// Wildcard `*` matches all array elements or object values.
    Wildcard,
    /// Filtered wildcard `*[pred]`; applies the op only to matching children.
    WildcardFilter(Box<Expr>),
    /// Recursive descent to all nodes named `field` at any depth.
    Descendant(String),
}


/// Target type for an `as` cast expression.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CastType {
    /// Cast to `i64`.
    Int,
    /// Cast to `f64`.
    Float,
    /// Cast to the most precise numeric type that fits.
    Number,
    /// Cast to string via `Display`.
    Str,
    /// Cast to boolean; empty / zero / null → false.
    Bool,
    /// Wrap scalar in a single-element array; pass arrays through.
    Array,
    /// Coerce to `Val::Obj`; null → empty object.
    Object,
    /// Always returns `Val::Null`; useful to conditionally erase a field.
    Null,
}


/// One stage in a `Pipeline` expression.
#[derive(Debug, Clone)]
pub enum PipeStep {
    /// Pass the current value through an expression (`| expr`).
    Forward(Expr),
    /// Destructure the current value into named bindings (`| as $name`).
    Bind(BindTarget),
}

/// Destructuring pattern used by a `PipeStep::Bind`.
#[derive(Debug, Clone)]
pub enum BindTarget {
    /// Bind the whole value to a single name (`as $x`).
    Name(String),
    /// Destructure an object into named fields with an optional rest capture.
    Obj {
        /// Field names extracted from the object.
        fields: Vec<String>,
        /// Optional name to capture remaining fields.
        rest: Option<String>,
    },
    /// Destructure an array positionally into named slots.
    Arr(Vec<String>),
}


/// One part of an `FString` template.
#[derive(Debug, Clone)]
pub enum FStringPart {
    /// A literal string segment between interpolation sites.
    Lit(String),
    /// An interpolated expression with an optional formatting directive.
    Interp { expr: Expr, fmt: Option<FmtSpec> },
}

/// Formatting directive attached to an FString interpolation site.
#[derive(Debug, Clone)]
pub enum FmtSpec {
    /// Python-style format spec string (e.g. `:.2f`).
    Spec(String),
    /// Named pipe formatter applied to the interpolated value.
    Pipe(String),
}


/// One element inside an array literal.
#[derive(Debug, Clone)]
pub enum ArrayElem {
    /// A single expression contributing one element.
    Expr(Expr),
    /// Spread operator `...expr`; splices an iterable's items inline.
    Spread(Expr),
}


/// Controls how `.first` / `.one` quantifiers resolve a multi-value result.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantifierKind {
    /// Return the first element, or null if the array is empty.
    First,
    /// Return the single element; error if the array has ≠ 1 element.
    One,
}

/// One postfix navigation step in a `Chain` expression.
#[derive(Debug, Clone)]
pub enum Step {
    /// `.field` — mandatory field access; propagates null if the key is absent.
    Field(String),
    /// `.field?` — optional field access; returns null without error when absent.
    OptField(String),
    /// `..field` — recursive descent collecting all nodes named `field`.
    Descendant(String),
    /// `..**` — recursive descent collecting every descendant.
    DescendAll,
    /// `[n]` — integer index; negative values count from the end.
    Index(i64),
    /// `[expr]` — runtime-computed index; `expr` is evaluated as a key or integer.
    DynIndex(Box<Expr>),
    /// `[start:end]` — array slice; either bound may be absent (open range).
    Slice(Option<i64>, Option<i64>),
    /// `.method(args…)` — method call dispatched through the builtin / custom registry.
    Method(String, Vec<Arg>),
    /// `.method?(args…)` — optional method call; errors become null.
    OptMethod(String, Vec<Arg>),
    /// `[pred]` — inline filter; keeps array elements for which `pred` is truthy.
    InlineFilter(Box<Expr>),
    /// `.first` / `.one` — quantifier that collapses an array to a scalar.
    Quantifier(QuantifierKind),
}


/// One argument in a method or global-function call.
#[derive(Debug, Clone)]
pub enum Arg {
    /// A positional argument.
    Pos(Expr),
    /// A named (keyword) argument.
    Named(String, Expr),
}


/// One field in an object literal.
#[derive(Debug, Clone)]
pub enum ObjField {
    /// A full `key: value` pair with optional omit-if-null and conditional flags.
    Kv {
        /// String key for this field.
        key: String,
        /// Value expression.
        val: Expr,
        /// When `true`, the field is omitted from the output if `val` evaluates to null.
        optional: bool,
        /// When present, the field is omitted unless this expression is truthy.
        cond: Option<Expr>,
    },
    /// Shorthand `{name}` — equivalent to `{name: $.name}` when parsed in context.
    Short(String),
    /// Computed key `{[key_expr]: val_expr}` — both key and value are evaluated at runtime.
    Dynamic { key: Expr, val: Expr },
    /// Shallow spread `{...expr}` — merges all key/value pairs of `expr` one level deep.
    Spread(Expr),
    /// Deep recursive spread `{**expr}` — recursively merges nested objects.
    SpreadDeep(Expr),
}


/// Binary infix operator. Variants map 1-to-1 to opcodes after compilation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinOp {
    /// Numeric addition or string concatenation.
    Add,
    /// Numeric subtraction.
    Sub,
    /// Numeric multiplication.
    Mul,
    /// Floating-point division; always returns `f64`.
    Div,
    /// Integer modulo.
    Mod,
    /// Structural equality; compares deeply for objects and arrays.
    Eq,
    /// Structural inequality.
    Neq,
    /// Strict less-than comparison.
    Lt,
    /// Less-than-or-equal comparison.
    Lte,
    /// Strict greater-than comparison.
    Gt,
    /// Greater-than-or-equal comparison.
    Gte,
    /// Fuzzy / substring match (`~=`).
    Fuzzy,
    /// Short-circuit logical AND; right side compiled into a sub-program.
    And,
    /// Short-circuit logical OR; right side compiled into a sub-program.
    Or,
}


/// Runtime type tag used with `is` / `is not` kind-check expressions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KindType {
    /// Matches `Val::Null`.
    Null,
    /// Matches `Val::Bool`.
    Bool,
    /// Matches any numeric variant (`Val::Int`, `Val::Float`).
    Number,
    /// Matches `Val::Str` and `Val::StrRef`.
    Str,
    /// Matches `Val::Arr`.
    Array,
    /// Matches `Val::Obj`.
    Object,
}

impl Expr {
    /// Wrap `self` in a `Chain` only when `steps` is non-empty; avoids
    /// spurious chain nodes for bare navigations with no postfix steps.
    pub fn maybe_chain(self, steps: Vec<Step>) -> Self {
        if steps.is_empty() {
            self
        } else {
            Expr::Chain(Box::new(self), steps)
        }
    }
}
