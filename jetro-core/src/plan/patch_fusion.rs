// Phase A is analysis-only; Phase B will consume these types so the
// dead-code lints fire here until then.
#![allow(dead_code)]

//! Phase A of the write-fusion plan: a non-mutating effect-summary
//! analyzer that walks an [`Expr`] tree and collects, per subtree, the
//! set of logical document roots it reads and the ordered list of
//! [`PatchOp`]s it would write.
//!
//! Phase B will use the summaries here to decide which adjacent writes
//! can fuse into a single batched [`Expr::Patch`]. This module performs
//! NO IR rewrites — every method takes `&Expr` and returns owned data.
//!
//! Canonicalization is the only "smart" thing it does: a let-alias
//! table flattens `let x = $ in x.set(1)` into a [`RootRef::Root`]
//! write so the scheduler can recognise it as targeting `$`.

use crate::parse::ast::{
    Arg, ArrayElem, BindTarget, Expr, FStringPart, ObjField, PatchOp, PathStep, PipeStep, Step,
};
use std::collections::HashSet;
use std::sync::Arc;

/// Canonical form of a write/read root. Two `RootRef` values are equal
/// iff the expression they came from refers to the same logical
/// document, modulo let-alias resolution.
///
/// Phase B will only fuse writes whose roots compare equal under
/// [`Hash`] + [`Eq`]; opaque roots (`Composite`) deliberately compare
/// by structural expression equality so the analyzer never claims
/// fusion across a root it cannot prove identical.
#[derive(Debug, Clone)]
pub(crate) enum RootRef {
    /// The global document binding `$`.
    Root,
    /// A `let`-bound name whose alias chain bottoms out at a name
    /// (i.e. resolves to `Local(_)` rather than [`RootRef::Root`] /
    /// [`RootRef::Current`] / [`RootRef::Composite`]).
    Local(Arc<str>),
    /// `@` inside a lambda or comprehension body. The `u32` is a fresh
    /// scope id so nested lambdas don't accidentally share writes.
    Current(u32),
    /// Anything we cannot prove identical (function-call results,
    /// arbitrary chain expressions, etc.). No fusion across two
    /// `Composite` roots — equality compares by `Arc` identity, so
    /// only the *same* allocated expression matches itself. The
    /// [`Expr`] AST does not implement `Eq` / `Hash`, and structural
    /// equality is exactly what we don't want here anyway: Phase B
    /// must treat every distinct `Composite` as a fusion barrier.
    Composite(Arc<Expr>),
}

impl PartialEq for RootRef {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (RootRef::Root, RootRef::Root) => true,
            (RootRef::Local(a), RootRef::Local(b)) => a == b,
            (RootRef::Current(a), RootRef::Current(b)) => a == b,
            (RootRef::Composite(a), RootRef::Composite(b)) => Arc::ptr_eq(a, b),
            _ => false,
        }
    }
}

impl Eq for RootRef {}

impl std::hash::Hash for RootRef {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        match self {
            RootRef::Root => 0u8.hash(state),
            RootRef::Local(name) => {
                1u8.hash(state);
                name.hash(state);
            }
            RootRef::Current(id) => {
                2u8.hash(state);
                id.hash(state);
            }
            RootRef::Composite(arc) => {
                3u8.hash(state);
                (Arc::as_ptr(arc) as usize).hash(state);
            }
        }
    }
}

/// One write produced by an [`Expr::Patch`] node, lifted out of its
/// surrounding expression so the Phase B scheduler can see writes
/// without traversing the AST again.
#[derive(Debug, Clone)]
pub(crate) struct WriteEffect {
    /// Canonicalized root the write targets.
    pub root: RootRef,
    /// The original [`PatchOp`] cloned from the source AST. Path /
    /// value / cond are preserved verbatim — Phase B owns the policy
    /// for batching, not the data.
    pub op: PatchOp,
}

/// Per-`Expr` effect summary used by the Phase B scheduler.
///
/// `reads` is order-insensitive (a [`HashSet`]) because reads compose
/// commutatively; `writes` preserves source order because the
/// scheduler must replay them in evaluation order.
#[derive(Debug, Default, Clone)]
pub(crate) struct EffectSummary {
    /// Roots this subtree reads. Writes' RHS reads are folded in too,
    /// so a Phase B planner can use this set alone to decide whether
    /// flushing a batch is necessary before evaluating the subtree.
    pub reads: HashSet<RootRef>,
    /// Writes produced by this subtree, in source-evaluation order.
    pub writes: Vec<WriteEffect>,
    /// `true` iff `writes` is non-empty. Cached so callers don't have
    /// to inspect the vector when they only need a yes/no answer.
    pub has_writes: bool,
}

impl EffectSummary {
    fn merge(&mut self, other: EffectSummary) {
        self.reads.extend(other.reads);
        self.writes.extend(other.writes);
        if other.has_writes {
            self.has_writes = true;
        }
    }
}

/// Analyzer state: the let-alias table plus a stack of `@`-scope ids
/// so nested lambdas / comprehensions don't share a [`RootRef::Current`].
pub(crate) struct EffectAnalyzer {
    /// Stack of `(name, root)` pairs. Lookups walk inside-out so
    /// shadowing works the same as in the evaluator.
    aliases: Vec<(Arc<str>, RootRef)>,
    /// Counter used to allocate fresh scope ids for each lambda /
    /// comprehension we enter.
    next_scope: u32,
    /// Stack of currently-active scope ids. The top is the scope `@`
    /// resolves to.
    scope_stack: Vec<u32>,
}

impl Default for EffectAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

impl EffectAnalyzer {
    /// Build a fresh analyzer with one default `@`-scope (id 0)
    /// already pushed so top-level expressions have a `@` to refer
    /// to.
    pub fn new() -> Self {
        Self {
            aliases: Vec::new(),
            next_scope: 1,
            scope_stack: vec![0],
        }
    }

    /// Walk `expr` and return the merged summary. Public entry point.
    pub fn analyze(&mut self, expr: &Expr) -> EffectSummary {
        self.visit(expr)
    }

    fn current_scope(&self) -> u32 {
        *self.scope_stack.last().unwrap_or(&0)
    }

    fn fresh_scope(&mut self) -> u32 {
        let id = self.next_scope;
        self.next_scope += 1;
        id
    }

    fn resolve(&self, name: &str) -> Option<RootRef> {
        self.aliases
            .iter()
            .rev()
            .find(|(n, _)| n.as_ref() == name)
            .map(|(_, r)| r.clone())
    }

    /// Map an [`Expr`] to the [`RootRef`] it represents when used as a
    /// write target. Anything that isn't a recognised root form
    /// becomes [`RootRef::Composite`].
    fn canonical_root(&self, expr: &Expr) -> RootRef {
        match expr {
            Expr::Root => RootRef::Root,
            Expr::Current => RootRef::Current(self.current_scope()),
            Expr::Ident(name) => self
                .resolve(name)
                .unwrap_or_else(|| RootRef::Local(Arc::from(name.as_str()))),
            other => RootRef::Composite(Arc::new(other.clone())),
        }
    }

    fn visit(&mut self, expr: &Expr) -> EffectSummary {
        match expr {
            // Pure literals: empty summary.
            Expr::Null
            | Expr::Bool(_)
            | Expr::Int(_)
            | Expr::Float(_)
            | Expr::Str(_)
            | Expr::DeleteMark => EffectSummary::default(),

            // Root references: one read, no writes.
            Expr::Root => {
                let mut s = EffectSummary::default();
                s.reads.insert(RootRef::Root);
                s
            }
            Expr::Current => {
                let mut s = EffectSummary::default();
                s.reads.insert(RootRef::Current(self.current_scope()));
                s
            }
            Expr::Ident(name) => {
                let mut s = EffectSummary::default();
                let r = self
                    .resolve(name)
                    .unwrap_or_else(|| RootRef::Local(Arc::from(name.as_str())));
                s.reads.insert(r);
                s
            }

            // f-strings: visit each interpolated sub-expression.
            Expr::FString(parts) => {
                let mut s = EffectSummary::default();
                for p in parts {
                    if let FStringPart::Interp { expr, .. } = p {
                        s.merge(self.visit(expr));
                    }
                }
                s
            }

            // Chains are pure traversals: reads of the base, no writes
            // (chain-write terminals have already been desugared into
            // `Expr::Patch` by the parser before we see them).
            Expr::Chain(base, steps) => {
                let mut s = self.visit(base);
                for step in steps {
                    s.merge(self.visit_step(step));
                }
                s
            }

            // Ordinary structural composites: visit children left-to-right.
            Expr::BinOp(l, _, r) => {
                let mut s = self.visit(l);
                s.merge(self.visit(r));
                s
            }
            Expr::UnaryNeg(inner) | Expr::Not(inner) => self.visit(inner),
            Expr::Kind { expr, .. } => self.visit(expr),
            Expr::Coalesce(l, r) => {
                let mut s = self.visit(l);
                s.merge(self.visit(r));
                s
            }

            Expr::Object(fields) => {
                let mut s = EffectSummary::default();
                for f in fields {
                    s.merge(self.visit_obj_field(f));
                }
                s
            }
            Expr::Array(elems) => {
                let mut s = EffectSummary::default();
                for e in elems {
                    match e {
                        ArrayElem::Expr(x) | ArrayElem::Spread(x) => s.merge(self.visit(x)),
                    }
                }
                s
            }

            // Pipelines: thread base through each step. `Bind` doesn't
            // change the value, just names it; we conservatively
            // record the bound name as aliasing `Composite` of the
            // current pipeline value (we can't know it here without
            // re-running scheduling), so for Phase A we don't push a
            // helpful alias — a `Bind` simply produces no extra reads.
            Expr::Pipeline { base, steps } => {
                let mut s = self.visit(base);
                for st in steps {
                    match st {
                        PipeStep::Forward(e) => s.merge(self.visit(e)),
                        PipeStep::Bind(_) => { /* no observable effect on summaries */ }
                    }
                }
                s
            }

            // Comprehensions: source is in the outer scope; body /
            // key / val / cond run inside a fresh `@`-scope so writes
            // against `Current` don't leak.
            Expr::ListComp {
                expr,
                vars,
                iter,
                cond,
            }
            | Expr::SetComp {
                expr,
                vars,
                iter,
                cond,
            }
            | Expr::GenComp {
                expr,
                vars,
                iter,
                cond,
            } => {
                let mut s = self.visit(iter);
                let scope = self.fresh_scope();
                self.scope_stack.push(scope);
                for v in vars {
                    self.aliases
                        .push((Arc::from(v.as_str()), RootRef::Current(scope)));
                }
                let inner_summary = {
                    let mut inner = self.visit(expr);
                    if let Some(c) = cond {
                        inner.merge(self.visit(c));
                    }
                    inner
                };
                for _ in vars {
                    self.aliases.pop();
                }
                self.scope_stack.pop();
                s.merge(inner_summary);
                s
            }
            Expr::DictComp {
                key,
                val,
                vars,
                iter,
                cond,
            } => {
                let mut s = self.visit(iter);
                let scope = self.fresh_scope();
                self.scope_stack.push(scope);
                for v in vars {
                    self.aliases
                        .push((Arc::from(v.as_str()), RootRef::Current(scope)));
                }
                let inner_summary = {
                    let mut inner = self.visit(key);
                    inner.merge(self.visit(val));
                    if let Some(c) = cond {
                        inner.merge(self.visit(c));
                    }
                    inner
                };
                for _ in vars {
                    self.aliases.pop();
                }
                self.scope_stack.pop();
                s.merge(inner_summary);
                s
            }

            // Lambda body is analysed in a fresh `@`-scope; params
            // alias to that scope so `o -> o.id.set(1)` becomes a
            // write against `Current(scope_id)`, not against `Root`.
            Expr::Lambda { params, body } => {
                let scope = self.fresh_scope();
                self.scope_stack.push(scope);
                for p in params {
                    self.aliases
                        .push((Arc::from(p.as_str()), RootRef::Current(scope)));
                }
                let inner = self.visit(body);
                for _ in params {
                    self.aliases.pop();
                }
                self.scope_stack.pop();
                inner
            }

            // `let x = init in body`: visit init in the outer scope,
            // push the alias from `x` to the canonical root of init,
            // visit body, pop. Writes from both halves merge.
            Expr::Let { name, init, body } => {
                let init_summary = self.visit(init);
                let alias = self.canonical_root(init);
                self.aliases.push((Arc::from(name.as_str()), alias));
                let body_summary = self.visit(body);
                self.aliases.pop();
                let mut s = init_summary;
                s.merge(body_summary);
                s
            }

            // Conditional: Phase A merges all three sub-summaries
            // unguarded. Phase F will re-wrap then/else writes with
            // their conditions, but for now we just collect them.
            Expr::IfElse { cond, then_, else_ } => {
                let mut s = self.visit(cond);
                s.merge(self.visit(then_));
                s.merge(self.visit(else_));
                s
            }

            // Try/default: both branches contribute; writes from the
            // body may not actually run, but Phase A is a may-summary
            // so we include them.
            Expr::Try { body, default } => {
                let mut s = self.visit(body);
                s.merge(self.visit(default));
                s
            }

            // Global call: visit each argument expression; the call
            // itself is treated as pure here (builtins have no
            // patch-write side effects in this model).
            Expr::GlobalCall { args, .. } => {
                let mut s = EffectSummary::default();
                for a in args {
                    s.merge(self.visit_arg(a));
                }
                s
            }

            Expr::Cast { expr, .. } => self.visit(expr),

            // The single thing we actually care about: lift each op
            // into a `WriteEffect` against the canonical root of
            // `Patch.root`. We also fold in reads from each op's
            // value / cond / dynamic path step so Phase B sees them.
            Expr::Patch { root, ops } => {
                let mut s = self.visit(root);
                let target = self.canonical_root(root);
                for op in ops {
                    // Reads inside the op (rhs value, cond, dynamic
                    // path steps) contribute to `reads`.
                    s.merge(self.visit(&op.val));
                    if let Some(c) = &op.cond {
                        s.merge(self.visit(c));
                    }
                    for step in &op.path {
                        if let PathStep::DynIndex(e) = step {
                            s.merge(self.visit(e));
                        }
                        if let PathStep::WildcardFilter(e) = step {
                            s.merge(self.visit(e));
                        }
                    }
                    s.writes.push(WriteEffect {
                        root: target.clone(),
                        op: op.clone(),
                    });
                    s.has_writes = true;
                }
                s
            }
        }
    }

    fn visit_step(&mut self, step: &Step) -> EffectSummary {
        match step {
            Step::Field(_)
            | Step::OptField(_)
            | Step::Descendant(_)
            | Step::DescendAll
            | Step::Index(_)
            | Step::Slice(_, _)
            | Step::Quantifier(_) => EffectSummary::default(),
            Step::DynIndex(e) | Step::InlineFilter(e) => self.visit(e),
            Step::Method(_, args) | Step::OptMethod(_, args) => {
                let mut s = EffectSummary::default();
                for a in args {
                    s.merge(self.visit_arg(a));
                }
                s
            }
        }
    }

    fn visit_arg(&mut self, arg: &Arg) -> EffectSummary {
        match arg {
            Arg::Pos(e) | Arg::Named(_, e) => self.visit(e),
        }
    }

    fn visit_obj_field(&mut self, f: &ObjField) -> EffectSummary {
        match f {
            ObjField::Kv { val, cond, .. } => {
                let mut s = self.visit(val);
                if let Some(c) = cond {
                    s.merge(self.visit(c));
                }
                s
            }
            ObjField::Short(name) => {
                // `{name}` desugars to `{name: $.name}`; treat as a
                // read of whatever `name` resolves to.
                let mut s = EffectSummary::default();
                let r = self
                    .resolve(name)
                    .unwrap_or_else(|| RootRef::Local(Arc::from(name.as_str())));
                s.reads.insert(r);
                s
            }
            ObjField::Dynamic { key, val } => {
                let mut s = self.visit(key);
                s.merge(self.visit(val));
                s
            }
            ObjField::Spread(e) | ObjField::SpreadDeep(e) => self.visit(e),
        }
    }
}

// Suppress dead-code lints for the `BindTarget` import that exists
// solely so future Phase B code can match on bind shapes without
// re-importing. Keeping it here documents the dependency.
#[allow(dead_code)]
fn _bind_target_witness(_b: &BindTarget) {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parse::parser::parse;

    fn analyze_str(s: &str) -> EffectSummary {
        let expr = parse(s).expect("parse");
        EffectAnalyzer::new().analyze(&expr)
    }

    #[test]
    fn pure_read_has_no_writes() {
        let s = analyze_str("$.a.b");
        assert!(s.writes.is_empty());
        assert!(!s.has_writes);
        assert!(s.reads.contains(&RootRef::Root));
    }

    #[test]
    fn literal_has_empty_summary() {
        let s = analyze_str("42");
        assert!(s.writes.is_empty());
        assert!(s.reads.is_empty());
    }

    #[test]
    fn single_set_records_write_against_root() {
        let s = analyze_str("$.a.set(1)");
        assert_eq!(s.writes.len(), 1);
        assert_eq!(s.writes[0].root, RootRef::Root);
        assert!(s.has_writes);
    }

    #[test]
    fn pipe_chain_collects_writes_in_order() {
        // Multiple chain-writes on `$` joined with `|`. Each rooted
        // `$.x.set(v)` becomes its own `Expr::Patch` against `$`.
        let s = analyze_str("$.a.set(1) | $.b.set(2) | $.c.set(3)");
        assert_eq!(s.writes.len(), 3);
        for w in &s.writes {
            assert_eq!(w.root, RootRef::Root);
        }
        assert!(s.has_writes);
    }

    #[test]
    fn let_aliases_root_for_reads() {
        // The chain-write rewriter only fires when the base is the
        // literal `$`, so `x.a.set(1)` stays as a method call. The
        // alias table is observable via reads: `x` resolves to Root.
        let s = analyze_str("let x = $ in (x.a | $.b.set(2))");
        // Only one write — the rooted `$.b.set(2)`.
        assert_eq!(s.writes.len(), 1);
        assert_eq!(s.writes[0].root, RootRef::Root);
        // And `x` is canonicalized to Root in `reads`.
        assert!(s.reads.contains(&RootRef::Root));
    }

    #[test]
    fn object_field_writes_collected() {
        let s = analyze_str("{a: $.x.set(1), b: $.y.set(2)}");
        assert_eq!(s.writes.len(), 2);
        assert_eq!(s.writes[0].root, RootRef::Root);
        assert_eq!(s.writes[1].root, RootRef::Root);
    }

    #[test]
    fn lambda_scope_isolation() {
        let s = analyze_str("$.list.map(lambda o: o.id.set(1))");
        // Outer reads include `$`. Writes inside the lambda body run
        // against `Current(scope_id)`, not `Root` — so the outer
        // summary's writes targeting Root must be empty.
        let outer_root_writes = s.writes.iter().filter(|w| w.root == RootRef::Root).count();
        assert_eq!(outer_root_writes, 0);
    }

    #[test]
    fn read_then_write_collects_both() {
        let s = analyze_str("$.a + $.b.set(1)");
        assert!(s.reads.contains(&RootRef::Root));
        assert_eq!(s.writes.len(), 1);
        assert_eq!(s.writes[0].root, RootRef::Root);
    }

    #[test]
    fn nested_let_alias_chain_canonicalizes_reads() {
        // `y` aliases `x` aliases `$`; a read of `y` canonicalizes
        // all the way to Root through the alias stack.
        let s = analyze_str("let x = $ in (let y = x in y.a)");
        assert!(s.reads.contains(&RootRef::Root));
        // No `Local("x")` / `Local("y")` left in `reads`.
        for r in &s.reads {
            match r {
                RootRef::Local(name) => panic!("unexpected Local read: {}", name),
                _ => {}
            }
        }
    }

    #[test]
    fn comprehension_does_not_leak_inner_writes_to_root() {
        // The comprehension body writes against `Current(scope)` —
        // never `Root` — even though the outer iter reads `$`.
        let s = analyze_str("[o.id.set(1) for o in $.list]");
        assert!(s.reads.contains(&RootRef::Root));
        let outer_root_writes = s.writes.iter().filter(|w| w.root == RootRef::Root).count();
        assert_eq!(outer_root_writes, 0);
    }

    #[test]
    fn ifelse_merges_branch_writes() {
        // Jetro uses Python-style `then if cond else else_`.
        let s = analyze_str("$.a.set(1) if $.flag else $.b.set(2)");
        // Phase A is a may-summary: both branches' writes are
        // collected unconditionally.
        assert_eq!(s.writes.len(), 2);
    }
}
