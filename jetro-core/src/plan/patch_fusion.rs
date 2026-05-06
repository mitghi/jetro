// Phase A types are analysis-only; Phase B's `fuse_writes` consumes them.
// Some helpers remain available for Phase C/E/F so we keep the lint
// allowance until those phases land.
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

// ---------------------------------------------------------------------------
// Phase B: contiguous same-root write fusion (IR rewrite)
// ---------------------------------------------------------------------------
//
// `fuse_writes` rewrites an [`Expr`] tree so adjacent writes targeting the
// same canonical root collapse into a single multi-op [`Expr::Patch`]. The
// downstream compiler builds one `CompiledPatch` per fused node, and Phase D's
// `CompiledPatchTrie::from_ops` then triages the multi-op forms onto the
// shared-`Arc::make_mut` execution path.
//
// Patterns handled here:
//
// * **P1** — `Pipeline { Forward, … }` chains with adjacent same-root writes.
//   The chain-write rewriter only fires for `$`-rooted bases, so a pipe
//   stage like `… | @.b.set(2)` parses as `Chain(Current, [..set])`. We
//   detect that shape during fusion (option (b) from the plan) and lift it
//   into a synthetic `Patch{root:@,…}` so the `RootRef::Current` adjacency
//   check below sees both writes against the same scope.
// * **P2** — `Object` field writes that share a canonical root: lifted
//   into a `let _patch_fuse_<n> = patch $ {…}` outside the object, with
//   each fused field replaced by a reference to the binding name. Other
//   fields keep their original expressions; we re-target rooted reads
//   inside *those* fields to read from the post-batch document so the
//   semantics match a sequential left-to-right evaluation order.
// * **P3** — `Let` whose `init` is a [`Expr::Patch`] and whose body
//   contains adjacent same-root writes through the alias name. The let
//   alias resolves to the patch's root (via Phase A's alias table); we
//   pull the body's writes into the init's op list and replace them with
//   references to the let-bound name.
// * **P4** — Nested pipelines: each pass merges any new same-root pair
//   exposed by the prior pass; we run a fixpoint loop until no rule fires.
//
// Soundness guards (matching `write_fusion_plan.md` invariants 1-7):
//
// * A pipeline stage that **reads** a root being batched against forces a
//   flush — we stop merging at that boundary.
// * Conditional ops (`PatchOp.cond.is_some()`) are not fused; they remain
//   in their original `Patch` so Phase F can handle them later.
// * Lambda / comprehension bodies introduce a fresh `Current` scope; their
//   writes never leak into an enclosing batch.
// * Source order is preserved: ops are concatenated in left-to-right
//   evaluation order so last-write-wins semantics line up with the
//   trie-builder's contract.
// * Composite roots never fuse (no fusion across opaque expressions).

/// Public entry: rewrite `expr` so contiguous same-root chain-writes
/// collapse into single multi-op [`Expr::Patch`] nodes. Pure: takes
/// ownership and returns the new tree.
///
/// Phase C wraps the result in a final `flush_all` so any pending
/// batches accumulated by the recursion bottom out as `let _N = patch …
/// in body`. In the current pass Phase B's local lifters return their
/// fused output directly so `pending` is typically empty here, but the
/// final flush is the safety net Phase E will lean on.
pub(crate) fn fuse_writes(expr: Expr) -> Expr {
    let mut ctx = FuseCtx::default();
    let body = fuse_recursive(expr, &mut ctx);
    ctx.flush_all(body)
}

/// Phase C: recurse into a child that lives behind a non-fuseable
/// scope boundary (lambda body, comprehension iter / body / cond,
/// if/else branch, try-default). The outer pending map is taken before
/// recursion so the child sees an empty `pending`; whatever batches
/// the child produces flush *inside* the child's returned expression
/// before we restore the outer state. Net effect: a write inside a
/// branch can never be merged with a write outside that branch.
fn fuse_subtree(child: Expr, ctx: &mut FuseCtx) -> Expr {
    let saved = ctx.take_pending();
    let inner = fuse_recursive(child, ctx);
    let inner = ctx.flush_all(inner);
    ctx.restore_pending(saved);
    inner
}

/// Maximum number of alias-chain hops [`FuseCtx::canonical_root`] will
/// chase before bailing to [`RootRef::Composite`]. Phase C's transitive
/// resolution can in principle iterate forever if the alias table
/// contains a cycle (`a -> b -> a`), which a well-formed `let` program
/// cannot produce — but a defensive cap means the optimizer can't be
/// tipped into a hang by malformed analyzer state.
const MAX_ALIAS_CHAIN_DEPTH: usize = 16;

/// One open batch of writes against a single canonical root, awaiting
/// flush. Phase C tracks these per-root in [`FuseCtx::pending`] so a
/// sequence of writes can collect into a single emitted `Patch` even
/// when the surrounding shape isn't directly handled by the Phase B
/// local lifters.
#[derive(Debug, Clone)]
struct PendingBatch {
    /// The root expression to attach to the emitted [`Expr::Patch`].
    /// We preserve the *original* root expression (not a canonicalised
    /// `RootRef`) so the lowered Patch keeps user-visible semantics.
    root_expr: Expr,
    /// The synthesized binding name `__patch_fuse_N` that any reference
    /// to the post-batch document will resolve to.
    binding: String,
    /// Ops collected against this batch in source order.
    ops: Vec<PatchOp>,
}

/// Mutable state threaded through the rewrite. Phase C extends Phase B's
/// alias table with a `pending` tracker (open batches per root) and a
/// depth-capped multi-level alias resolver.
#[derive(Default)]
struct FuseCtx {
    /// Mirrors [`EffectAnalyzer::aliases`] for the duration of the
    /// rewrite so let-aliases compose with the canonical-root rule.
    aliases: Vec<(Arc<str>, RootRef)>,
    /// Stack of `Current` scope ids (lambda / comprehension boundaries).
    scope_stack: Vec<u32>,
    /// Scope id allocator.
    next_scope: u32,
    /// Counter for synthetic `let` names introduced by P2 (object lift)
    /// and Phase C's pending-batch flush.
    next_synth: u32,
    /// Phase C: open batches per canonical root, keyed in source order.
    /// `IndexMap` (not `HashMap`) so flush order matches insertion order
    /// — a deterministic emission shape is required by the soundness
    /// invariants on read-after-write coherence.
    pending: indexmap::IndexMap<RootRef, PendingBatch>,
}

impl FuseCtx {
    fn current_scope(&self) -> u32 {
        *self.scope_stack.last().unwrap_or(&0)
    }
    fn fresh_scope(&mut self) -> u32 {
        let id = self.next_scope;
        self.next_scope += 1;
        id
    }
    fn fresh_synth_name(&mut self) -> String {
        let id = self.next_synth;
        self.next_synth += 1;
        format!("__patch_fuse_{}", id)
    }
    fn resolve(&self, name: &str) -> Option<RootRef> {
        self.aliases
            .iter()
            .rev()
            .find(|(n, _)| n.as_ref() == name)
            .map(|(_, r)| r.clone())
    }

    /// Canonicalise an expression to a [`RootRef`].
    ///
    /// Phase C upgrades this to chase alias chains transitively: if the
    /// resolve step returns `Local(name)` and `name` is itself bound,
    /// we keep walking. The walk is bounded by [`MAX_ALIAS_CHAIN_DEPTH`]
    /// hops and breaks on cycles (where the same `Local` name is
    /// revisited), bailing to [`RootRef::Composite`] in either failure
    /// mode so the caller never observes a half-resolved alias.
    fn canonical_root(&self, expr: &Expr) -> RootRef {
        match expr {
            Expr::Root => RootRef::Root,
            Expr::Current => RootRef::Current(self.current_scope()),
            Expr::Ident(name) => self.canonical_local_chain(name.as_str()),
            other => RootRef::Composite(Arc::new(other.clone())),
        }
    }

    /// Resolve a name through the alias table, chasing transitive
    /// `Local -> Local` hops. Returns the deepest non-`Local` root we
    /// can prove, or `Local(seed)` if the name itself isn't in the
    /// table. On cycle / depth-cap violation we return
    /// [`RootRef::Composite`] of the original [`Expr::Ident`] so no
    /// fusion can falsely claim cross-root identity.
    fn canonical_local_chain(&self, seed: &str) -> RootRef {
        let mut current_name: Arc<str> = Arc::from(seed);
        let mut visited: Vec<Arc<str>> = Vec::new();
        for _ in 0..MAX_ALIAS_CHAIN_DEPTH {
            // Cycle guard: if we've already seen this name, fall back.
            if visited.iter().any(|v| v == &current_name) {
                return RootRef::Composite(Arc::new(Expr::Ident(seed.to_string())));
            }
            visited.push(current_name.clone());
            match self.resolve(&current_name) {
                None => return RootRef::Local(current_name),
                Some(RootRef::Local(next)) => {
                    // Keep chasing — the next hop might bottom out at
                    // Root / Current / Composite.
                    current_name = next;
                    continue;
                }
                Some(other) => return other,
            }
        }
        // Depth cap: defensive bail.
        RootRef::Composite(Arc::new(Expr::Ident(seed.to_string())))
    }

    /// Build a [`EffectAnalyzer`] seeded with our current alias / scope
    /// state so soundness checks (`reads.contains(...)`) compose.
    fn analyzer(&self) -> EffectAnalyzer {
        let mut a = EffectAnalyzer::new();
        a.aliases = self.aliases.clone();
        a.scope_stack = self.scope_stack.clone();
        // Make sure `next_scope` doesn't collide with ours.
        a.next_scope = self.next_scope;
        a
    }

    /// Phase C: append `ops` to the pending batch keyed by `root`.
    /// Creates a fresh [`PendingBatch`] (with a `__patch_fuse_N` binding
    /// name) when this is the first batch against the root. Returns the
    /// binding name; callers use it to construct an [`Expr::Ident`]
    /// placeholder where the post-batch value would have appeared.
    fn add_to_batch(&mut self, root: RootRef, root_expr: Expr, ops: Vec<PatchOp>) -> String {
        if let Some(batch) = self.pending.get_mut(&root) {
            batch.ops.extend(ops);
            batch.binding.clone()
        } else {
            let binding = self.fresh_synth_name();
            let batch = PendingBatch {
                root_expr,
                binding: binding.clone(),
                ops,
            };
            self.pending.insert(root, batch);
            binding
        }
    }

    /// Phase C: flush every pending batch, wrapping `body` in a
    /// `let __patch_fuse_N = Patch{...} in ...` chain. Insertion order
    /// of `pending` determines the wrapping order; the outermost let
    /// is the first batch added, matching source-order semantics.
    fn flush_all(&mut self, body: Expr) -> Expr {
        let mut wrapped = body;
        // Drain in reverse so the first-inserted batch is the outermost
        // let — preserves source order for nested reads.
        let drained: Vec<(RootRef, PendingBatch)> = self.pending.drain(..).collect();
        for (_root, batch) in drained.into_iter().rev() {
            wrapped = Expr::Let {
                name: batch.binding,
                init: Box::new(Expr::Patch {
                    root: Box::new(batch.root_expr),
                    ops: batch.ops,
                }),
                body: Box::new(wrapped),
            };
        }
        wrapped
    }

    /// Phase C: flush only the pending batch keyed by `root`. Used when
    /// a read of `root` is encountered; we must materialise the writes
    /// before the read evaluates.
    fn flush_root(&mut self, root: &RootRef, body: Expr) -> Expr {
        if let Some(batch) = self.pending.shift_remove(root) {
            Expr::Let {
                name: batch.binding,
                init: Box::new(Expr::Patch {
                    root: Box::new(batch.root_expr),
                    ops: batch.ops,
                }),
                body: Box::new(body),
            }
        } else {
            body
        }
    }

    /// Phase C: snapshot/restore for scope boundaries. Returns the
    /// pending map at the call site and replaces it with an empty one
    /// so children can collect their own batches without disturbing
    /// outer state.
    fn take_pending(&mut self) -> indexmap::IndexMap<RootRef, PendingBatch> {
        std::mem::take(&mut self.pending)
    }

    /// Phase C: restore a previously taken pending map.
    fn restore_pending(&mut self, prev: indexmap::IndexMap<RootRef, PendingBatch>) {
        self.pending = prev;
    }
}

/// Walk the expression bottom-up, fusing where shapes match. Most
/// variants only need to recurse into children — the interesting work
/// happens at `Pipeline`, `Object`, and `Let`.
fn fuse_recursive(expr: Expr, ctx: &mut FuseCtx) -> Expr {
    match expr {
        // Leaves and trivially recursive cases.
        Expr::Null
        | Expr::Bool(_)
        | Expr::Int(_)
        | Expr::Float(_)
        | Expr::Str(_)
        | Expr::Root
        | Expr::Current
        | Expr::Ident(_)
        | Expr::DeleteMark => expr,

        Expr::FString(parts) => Expr::FString(
            parts
                .into_iter()
                .map(|p| match p {
                    crate::parse::ast::FStringPart::Lit(s) => {
                        crate::parse::ast::FStringPart::Lit(s)
                    }
                    crate::parse::ast::FStringPart::Interp { expr, fmt } => {
                        crate::parse::ast::FStringPart::Interp {
                            expr: fuse_recursive(expr, ctx),
                            fmt,
                        }
                    }
                })
                .collect(),
        ),

        Expr::Chain(base, steps) => {
            let base = Box::new(fuse_recursive(*base, ctx));
            let steps = steps.into_iter().map(|s| fuse_step(s, ctx)).collect();
            Expr::Chain(base, steps)
        }

        Expr::BinOp(l, op, r) => Expr::BinOp(
            Box::new(fuse_recursive(*l, ctx)),
            op,
            Box::new(fuse_recursive(*r, ctx)),
        ),
        Expr::UnaryNeg(inner) => Expr::UnaryNeg(Box::new(fuse_recursive(*inner, ctx))),
        Expr::Not(inner) => Expr::Not(Box::new(fuse_recursive(*inner, ctx))),
        Expr::Kind { expr, ty, negate } => Expr::Kind {
            expr: Box::new(fuse_recursive(*expr, ctx)),
            ty,
            negate,
        },
        Expr::Coalesce(l, r) => Expr::Coalesce(
            Box::new(fuse_recursive(*l, ctx)),
            Box::new(fuse_recursive(*r, ctx)),
        ),

        Expr::Object(fields) => fuse_object(fields, ctx),
        Expr::Array(elems) => Expr::Array(
            elems
                .into_iter()
                .map(|e| match e {
                    ArrayElem::Expr(x) => ArrayElem::Expr(fuse_recursive(x, ctx)),
                    ArrayElem::Spread(x) => ArrayElem::Spread(fuse_recursive(x, ctx)),
                })
                .collect(),
        ),

        Expr::Pipeline { base, steps } => fuse_pipeline(*base, steps, ctx),

        Expr::ListComp { expr, vars, iter, cond } => {
            // Phase C: comprehension source runs in the outer scope but
            // each iter step is a new boundary; flush both before / after
            // entering the inner scope so cross-boundary leaks are
            // impossible. (Inner-scope writes today are already isolated
            // by the per-iter `Current` scope id; the seal here pins the
            // invariant in case future Phase E fusion adds cross-iter
            // batching.)
            let iter = Box::new(fuse_subtree(*iter, ctx));
            with_lambda_scope(ctx, &vars, |ctx| Expr::ListComp {
                expr: Box::new(fuse_subtree(*expr, ctx)),
                vars: vars.clone(),
                iter,
                cond: cond.map(|c| Box::new(fuse_subtree(*c, ctx))),
            })
        }
        Expr::SetComp { expr, vars, iter, cond } => {
            let iter = Box::new(fuse_subtree(*iter, ctx));
            with_lambda_scope(ctx, &vars, |ctx| Expr::SetComp {
                expr: Box::new(fuse_subtree(*expr, ctx)),
                vars: vars.clone(),
                iter,
                cond: cond.map(|c| Box::new(fuse_subtree(*c, ctx))),
            })
        }
        Expr::GenComp { expr, vars, iter, cond } => {
            let iter = Box::new(fuse_subtree(*iter, ctx));
            with_lambda_scope(ctx, &vars, |ctx| Expr::GenComp {
                expr: Box::new(fuse_subtree(*expr, ctx)),
                vars: vars.clone(),
                iter,
                cond: cond.map(|c| Box::new(fuse_subtree(*c, ctx))),
            })
        }
        Expr::DictComp { key, val, vars, iter, cond } => {
            let iter = Box::new(fuse_subtree(*iter, ctx));
            with_lambda_scope(ctx, &vars, |ctx| Expr::DictComp {
                key: Box::new(fuse_subtree(*key, ctx)),
                val: Box::new(fuse_subtree(*val, ctx)),
                vars: vars.clone(),
                iter,
                cond: cond.map(|c| Box::new(fuse_subtree(*c, ctx))),
            })
        }

        Expr::Lambda { params, body } => with_lambda_scope(ctx, &params, |ctx| Expr::Lambda {
            params: params.clone(),
            // Phase C: lambda body is a non-fuseable boundary — flush
            // both outer pending and the body's own pending so writes
            // never leak in either direction across the call edge.
            body: Box::new(fuse_subtree(*body, ctx)),
        }),

        Expr::Let { name, init, body } => fuse_let(name, *init, *body, ctx),

        Expr::IfElse { cond, then_, else_ } => Expr::IfElse {
            // Phase C: each branch is its own scope. A write that only
            // appears in `then_` must not be batched with one in `else_`,
            // since at most one branch runs. `fuse_subtree` flushes
            // pending at each boundary so any optimisation lifted into
            // a branch stays inside it.
            cond: Box::new(fuse_subtree(*cond, ctx)),
            then_: Box::new(fuse_subtree(*then_, ctx)),
            else_: Box::new(fuse_subtree(*else_, ctx)),
        },
        Expr::Try { body, default } => Expr::Try {
            // Phase C: a try-default boundary is non-fuseable; the body
            // may abort and the default observes the pre-failure state.
            body: Box::new(fuse_subtree(*body, ctx)),
            default: Box::new(fuse_subtree(*default, ctx)),
        },
        Expr::GlobalCall { name, args } => Expr::GlobalCall {
            name,
            args: args
                .into_iter()
                .map(|a| match a {
                    Arg::Pos(e) => Arg::Pos(fuse_recursive(e, ctx)),
                    Arg::Named(n, e) => Arg::Named(n, fuse_recursive(e, ctx)),
                })
                .collect(),
        },
        Expr::Cast { expr, ty } => Expr::Cast {
            expr: Box::new(fuse_recursive(*expr, ctx)),
            ty,
        },

        Expr::Patch { root, ops } => {
            let root = Box::new(fuse_recursive(*root, ctx));
            let ops = ops
                .into_iter()
                .map(|op| fuse_patch_op(op, ctx))
                .collect();
            Expr::Patch { root, ops }
        }
    }
}

fn fuse_step(step: Step, ctx: &mut FuseCtx) -> Step {
    match step {
        Step::DynIndex(e) => Step::DynIndex(Box::new(fuse_recursive(*e, ctx))),
        Step::InlineFilter(e) => Step::InlineFilter(Box::new(fuse_recursive(*e, ctx))),
        Step::Method(n, args) => Step::Method(
            n,
            args.into_iter()
                .map(|a| match a {
                    Arg::Pos(e) => Arg::Pos(fuse_recursive(e, ctx)),
                    Arg::Named(n, e) => Arg::Named(n, fuse_recursive(e, ctx)),
                })
                .collect(),
        ),
        Step::OptMethod(n, args) => Step::OptMethod(
            n,
            args.into_iter()
                .map(|a| match a {
                    Arg::Pos(e) => Arg::Pos(fuse_recursive(e, ctx)),
                    Arg::Named(n, e) => Arg::Named(n, fuse_recursive(e, ctx)),
                })
                .collect(),
        ),
        other => other,
    }
}

fn fuse_patch_op(op: PatchOp, ctx: &mut FuseCtx) -> PatchOp {
    let PatchOp { path, val, cond } = op;
    let path = path
        .into_iter()
        .map(|s| match s {
            PathStep::DynIndex(e) => PathStep::DynIndex(fuse_recursive(e, ctx)),
            PathStep::WildcardFilter(e) => {
                PathStep::WildcardFilter(Box::new(fuse_recursive(*e, ctx)))
            }
            other => other,
        })
        .collect();
    PatchOp {
        path,
        val: fuse_recursive(val, ctx),
        cond: cond.map(|c| fuse_recursive(c, ctx)),
    }
}

/// Helper to enter a lambda-/comprehension-scope: pushes a fresh
/// `Current` scope id and aliases each parameter to it, runs `f`, then
/// pops.
fn with_lambda_scope<R>(
    ctx: &mut FuseCtx,
    params: &[String],
    f: impl FnOnce(&mut FuseCtx) -> R,
) -> R {
    let scope = ctx.fresh_scope();
    ctx.scope_stack.push(scope);
    for p in params {
        ctx.aliases
            .push((Arc::from(p.as_str()), RootRef::Current(scope)));
    }
    let out = f(ctx);
    for _ in params {
        ctx.aliases.pop();
    }
    ctx.scope_stack.pop();
    out
}

// --- Pattern P1: Pipeline fusion ------------------------------------------

/// Try to lift a pipe-stage expression into an `Expr::Patch`.
///
/// The chain-write rewriter in the parser only fires for `$`-rooted
/// bases. Pipe stages like `… | @.b.set(2)` parse as
/// `Chain(Current, [Field("b"), Method("set", [arg])])`. This helper
/// recognises that exact shape and emits a synthetic `Patch{root:@,…}`
/// so the pipeline fuser can treat it uniformly.
///
/// Returns `None` when the stage is not a chain-write terminal — leaves
/// the expression untouched. Conservative: only fires when the base is
/// `Expr::Current` or `Expr::Ident` (lifting alias-equivalent let names
/// into Patches against the alias name; canonicalisation happens later).
fn lift_chain_write_pipe_stage(stage: Expr) -> Result<Expr, Expr> {
    let (base, steps) = match stage {
        Expr::Chain(b, s) => (*b, s),
        other => return Err(other),
    };
    // Must be a write-shaped chain.
    let last = match steps.last() {
        Some(s) => s,
        None => return Err(Expr::Chain(Box::new(base), steps)),
    };
    let (name, args) = match last {
        Step::Method(n, a) => (n.clone(), a.clone()),
        _ => return Err(Expr::Chain(Box::new(base), steps)),
    };
    if !is_write_terminal(&name) {
        return Err(Expr::Chain(Box::new(base), steps));
    }
    // Base must be a recognised root form.
    let allow = matches!(&base, Expr::Current | Expr::Ident(_) | Expr::Root);
    if !allow {
        return Err(Expr::Chain(Box::new(base), steps));
    }
    let prefix: Vec<Step> = steps[..steps.len() - 1].to_vec();
    let path = match steps_to_path(&prefix) {
        Some(p) => p,
        None => return Err(Expr::Chain(Box::new(base), steps)),
    };
    let op = match build_write_patch_op(&name, &args, path) {
        Some(o) => o,
        None => return Err(Expr::Chain(Box::new(base), steps)),
    };
    Ok(Expr::Patch {
        root: Box::new(base),
        ops: vec![op],
    })
}

/// Mirrors `parser::is_terminal_write` — duplicated locally so we don't
/// need to make that helper `pub(crate)` for one call site.
fn is_write_terminal(name: &str) -> bool {
    matches!(
        name,
        "set" | "modify" | "delete" | "unset" | "merge" | "deep_merge" | "deepMerge"
    )
}

fn steps_to_path(steps: &[Step]) -> Option<Vec<PathStep>> {
    let mut out = Vec::with_capacity(steps.len());
    for s in steps {
        match s {
            Step::Field(f) | Step::OptField(f) => out.push(PathStep::Field(f.clone())),
            Step::Index(i) => out.push(PathStep::Index(*i)),
            Step::Descendant(f) => out.push(PathStep::Descendant(f.clone())),
            Step::DynIndex(e) => out.push(PathStep::DynIndex((**e).clone())),
            _ => return None,
        }
    }
    Some(out)
}

/// Mirrors `parser::build_write_op`. We only need `set` / `modify` /
/// `delete` / `unset` for pipe-stage lifting; `merge` and `deep_merge`
/// already pass through the original parser path when the base is `$`,
/// and the pipe form `| .merge(x)` is unusual enough that we leave it
/// alone for now.
fn build_write_patch_op(
    name: &str,
    args: &[Arg],
    path: Vec<PathStep>,
) -> Option<PatchOp> {
    match name {
        "set" => {
            let v = arg_expr_owned(args.first()?);
            Some(PatchOp {
                path,
                val: v,
                cond: None,
            })
        }
        "modify" => {
            let v = match arg_expr_owned(args.first()?) {
                Expr::Lambda { params, body } => {
                    if let Some(p) = params.into_iter().next() {
                        Expr::Let {
                            name: p,
                            init: Box::new(Expr::Current),
                            body,
                        }
                    } else {
                        *body
                    }
                }
                other => other,
            };
            Some(PatchOp {
                path,
                val: v,
                cond: None,
            })
        }
        "delete" => {
            if !args.is_empty() {
                return None;
            }
            Some(PatchOp {
                path,
                val: Expr::DeleteMark,
                cond: None,
            })
        }
        "unset" => {
            let key = match arg_expr_owned(args.first()?) {
                Expr::Str(s) => s,
                Expr::Ident(s) => s,
                _ => return None,
            };
            let mut p = path;
            p.push(PathStep::Field(key));
            Some(PatchOp {
                path: p,
                val: Expr::DeleteMark,
                cond: None,
            })
        }
        _ => None,
    }
}

fn arg_expr_owned(a: &Arg) -> Expr {
    match a {
        Arg::Pos(e) | Arg::Named(_, e) => e.clone(),
    }
}

/// Fuse a pipeline. Walks left-to-right, merging adjacent same-root
/// `Patch` stages into the running base, splitting on:
///
/// * `PipeStep::Bind` (introduces a name in the surface; conservative
///   stop point so the binding's value is unambiguous)
/// * a stage that reads the root being batched (read-after-write barrier)
/// * a stage with non-fuseable shape
fn fuse_pipeline(base: Expr, steps: Vec<PipeStep>, ctx: &mut FuseCtx) -> Expr {
    // Recurse into base and each forward stage first.
    let mut acc: Expr = fuse_recursive(base, ctx);
    let mut new_steps: Vec<PipeStep> = Vec::with_capacity(steps.len());

    for st in steps {
        match st {
            PipeStep::Bind(t) => {
                // Flush — keep the bind as-is, accept the batched acc.
                new_steps.push(PipeStep::Bind(t));
            }
            PipeStep::Forward(stage_expr) => {
                let stage_expr = fuse_recursive(stage_expr, ctx);
                // Try to lift method-call form into a Patch first.
                let stage_expr = match lift_chain_write_pipe_stage(stage_expr) {
                    Ok(p) => p,
                    Err(other) => other,
                };

                if let Some(merged) = try_merge_pipeline_stage(&acc, &new_steps, &stage_expr, ctx) {
                    acc = merged;
                } else {
                    new_steps.push(PipeStep::Forward(stage_expr));
                }
            }
        }
    }

    if new_steps.is_empty() {
        acc
    } else {
        Expr::Pipeline {
            base: Box::new(acc),
            steps: new_steps,
        }
    }
}

/// Try to fuse `stage` into `acc` (the running pipeline base).
/// Returns `Some(new_base)` when the merge succeeded; `None` to fall
/// back to keeping the stage as a forward step.
fn try_merge_pipeline_stage(
    acc: &Expr,
    pending_steps: &[PipeStep],
    stage: &Expr,
    ctx: &FuseCtx,
) -> Option<Expr> {
    // Phase B only fuses when no forward steps have been buffered yet
    // (i.e. acc is still our cumulative batched Patch). Once a non-
    // fuseable stage lands in `pending_steps` we stop merging into acc.
    if !pending_steps.is_empty() {
        return None;
    }

    // Both sides must be `Patch` nodes.
    let (acc_root, acc_ops) = match acc {
        Expr::Patch { root, ops } => (root.as_ref(), ops),
        _ => return None,
    };
    let (stage_root, stage_ops) = match stage {
        Expr::Patch { root, ops } => (root.as_ref(), ops),
        _ => return None,
    };

    // Conditional ops disable trie fusion. Phase F handles them; for
    // now we leave both Patches separate so the trie path keeps its
    // simple invariants.
    if acc_ops.iter().any(|o| o.cond.is_some())
        || stage_ops.iter().any(|o| o.cond.is_some())
    {
        return None;
    }

    // Composite roots never fuse.
    let acc_root_ref = ctx.canonical_root(acc_root);
    if matches!(acc_root_ref, RootRef::Composite(_)) {
        return None;
    }

    // The stage's root, after canonicalisation, must equal the acc's
    // root OR `Current` — `@` in a pipe stage refers to the previous
    // stage's value, which by induction is `acc` (one batched Patch
    // against `acc_root`). So a `Patch{root:@,…}` against a pipeline
    // whose head writes to `acc_root` is targeting the same logical
    // document.
    let stage_root_ref = ctx.canonical_root(stage_root);
    let same_root = stage_root_ref == acc_root_ref
        || stage_root_ref == RootRef::Current(ctx.current_scope());
    if !same_root {
        return None;
    }

    // Soundness: the stage's ops' values / conds / dyn-indices must not
    // READ the root being written. We check ops directly rather than
    // analysing the whole stage `Expr::Patch`, because `Patch.root` is
    // (necessarily) a read of the target root and would always trigger
    // a false positive.
    let mut a = ctx.analyzer();
    for op in stage_ops {
        let val_summary = a.analyze(&op.val);
        if val_summary.reads.contains(&acc_root_ref) {
            return None;
        }
        if let Some(c) = &op.cond {
            if a.analyze(c).reads.contains(&acc_root_ref) {
                return None;
            }
        }
        for s in &op.path {
            if let PathStep::DynIndex(e) = s {
                if a.analyze(e).reads.contains(&acc_root_ref) {
                    return None;
                }
            }
            if let PathStep::WildcardFilter(e) = s {
                if a.analyze(e).reads.contains(&acc_root_ref) {
                    return None;
                }
            }
        }
    }

    // Concatenate ops in source order. The stage's ops were authored
    // against `Current` (= acc) but their paths and values are
    // structurally independent of which root expression we use —
    // the trie applies them to whatever document the merged Patch
    // ultimately walks. Last-write-wins is preserved by the
    // CompiledPatchTrie builder's left-to-right traversal.
    let mut merged_ops = acc_ops.clone();
    merged_ops.extend(stage_ops.iter().cloned());

    Some(Expr::Patch {
        root: Box::new(acc_root.clone()),
        ops: merged_ops,
    })
}

// --- Pattern P2: Object-field write fusion --------------------------------

/// Lift sibling Patches inside an object literal into a `let` binding
/// before the object so the doc materialises once. Conservative: only
/// fires when ≥ 2 same-canonical-root Patch fields exist back-to-back
/// AND no other field reads the same root (else lifting would change
/// the read-vs-write order).
///
/// Example transform:
///
/// ```text
/// {a: $.x.set(1), b: $.y.set(2), c: 3}
///   ↓
/// let __patch_fuse_0 = patch $ { x: 1, y: 2 } in
/// {a: __patch_fuse_0, b: __patch_fuse_0, c: 3}
/// ```
fn fuse_object(fields: Vec<ObjField>, ctx: &mut FuseCtx) -> Expr {
    // First, recursively fuse each field's sub-expressions.
    let recursed: Vec<ObjField> = fields
        .into_iter()
        .map(|f| match f {
            ObjField::Kv {
                key,
                val,
                optional,
                cond,
            } => ObjField::Kv {
                key,
                val: fuse_recursive(val, ctx),
                optional,
                cond: cond.map(|c| fuse_recursive(c, ctx)),
            },
            ObjField::Short(n) => ObjField::Short(n),
            ObjField::Dynamic { key, val } => ObjField::Dynamic {
                key: fuse_recursive(key, ctx),
                val: fuse_recursive(val, ctx),
            },
            ObjField::Spread(e) => ObjField::Spread(fuse_recursive(e, ctx)),
            ObjField::SpreadDeep(e) => ObjField::SpreadDeep(fuse_recursive(e, ctx)),
        })
        .collect();

    // Detect ≥ 2 Kv fields whose value is a Patch with a non-Composite
    // root — and whose roots all agree.
    let mut patch_indices: Vec<usize> = Vec::new();
    let mut shared_root: Option<RootRef> = None;
    let mut shared_root_expr: Option<Expr> = None;
    for (i, f) in recursed.iter().enumerate() {
        if let ObjField::Kv {
            val: Expr::Patch { root, ops },
            cond: None,
            ..
        } = f
        {
            if ops.iter().any(|o| o.cond.is_some()) {
                continue;
            }
            let r = ctx.canonical_root(root);
            if matches!(r, RootRef::Composite(_)) {
                continue;
            }
            match &shared_root {
                None => {
                    shared_root = Some(r.clone());
                    shared_root_expr = Some((**root).clone());
                    patch_indices.push(i);
                }
                Some(existing) if existing == &r => {
                    patch_indices.push(i);
                }
                _ => {
                    // Different root — can't fuse with the others; keep
                    // the first cluster only. (Phase C may relax this.)
                }
            }
        }
    }

    if patch_indices.len() < 2 {
        return Expr::Object(recursed);
    }
    let shared_root = shared_root.unwrap();
    let shared_root_expr = shared_root_expr.unwrap();

    // Soundness: no other field may read the root being batched.
    // Otherwise the field would observe pre-batch state in Jetro's
    // current evaluation order, but post-batch state after the lift.
    for (i, f) in recursed.iter().enumerate() {
        if patch_indices.contains(&i) {
            continue;
        }
        let summary = ctx.analyzer().analyze_obj_field(f);
        if summary.reads.contains(&shared_root) {
            return Expr::Object(recursed);
        }
        // Conservatively also bail if a non-fused field writes the
        // same root — preserves source-order semantics.
        if summary.writes.iter().any(|w| w.root == shared_root) {
            return Expr::Object(recursed);
        }
    }

    // Build the merged op list (concatenated in source order — using
    // `patch_indices` which is already increasing by construction).
    let mut merged_ops: Vec<PatchOp> = Vec::new();
    for &i in &patch_indices {
        if let ObjField::Kv {
            val: Expr::Patch { ops, .. },
            ..
        } = &recursed[i]
        {
            merged_ops.extend(ops.iter().cloned());
        }
    }

    let synth_name = ctx.fresh_synth_name();
    let synth_arc: Arc<str> = Arc::from(synth_name.as_str());

    // Replace each fused field's value with a reference to the synth
    // binding. Other fields stay unchanged — they didn't read the
    // shared root so they observe the same document either way.
    let new_fields: Vec<ObjField> = recursed
        .into_iter()
        .enumerate()
        .map(|(i, f)| {
            if patch_indices.contains(&i) {
                if let ObjField::Kv {
                    key,
                    optional,
                    cond,
                    ..
                } = f
                {
                    ObjField::Kv {
                        key,
                        val: Expr::Ident(synth_name.clone()),
                        optional,
                        cond,
                    }
                } else {
                    f
                }
            } else {
                f
            }
        })
        .collect();

    let object_expr = Expr::Object(new_fields);

    // Wrap in `let synth = patch shared_root { merged_ops } in object_expr`.
    // Push the alias so any nested `Ident(synth)` references are visible
    // — in practice we just inserted them, so the alias is conservative.
    ctx.aliases
        .push((synth_arc, shared_root.clone()));
    let body = object_expr;
    ctx.aliases.pop();

    Expr::Let {
        name: synth_name,
        init: Box::new(Expr::Patch {
            root: Box::new(shared_root_expr),
            ops: merged_ops,
        }),
        body: Box::new(body),
    }
}

// --- Pattern P3: Let init→body fusion -------------------------------------

/// Lift `Chain(<base>, [..steps, MethodWrite])` into a synthetic Patch.
/// Like `lift_chain_write_pipe_stage` but does not consume ownership;
/// returns `None` when the chain isn't a write terminal.
fn try_lift_chain_write(expr: &Expr) -> Option<Expr> {
    let (base, steps) = match expr {
        Expr::Chain(b, s) => (b.as_ref(), s.as_slice()),
        _ => return None,
    };
    let last = steps.last()?;
    let (name, args) = match last {
        Step::Method(n, a) => (n.clone(), a.clone()),
        _ => return None,
    };
    if !is_write_terminal(&name) {
        return None;
    }
    if !matches!(base, Expr::Current | Expr::Ident(_) | Expr::Root) {
        return None;
    }
    let prefix: Vec<Step> = steps[..steps.len() - 1].to_vec();
    let path = steps_to_path(&prefix)?;
    let op = build_write_patch_op(&name, &args, path)?;
    Some(Expr::Patch {
        root: Box::new(base.clone()),
        ops: vec![op],
    })
}

fn fuse_let(name: String, init: Expr, body: Expr, ctx: &mut FuseCtx) -> Expr {
    let init = fuse_recursive(init, ctx);
    let alias = ctx.canonical_root(&init_target(&init));
    ctx.aliases.push((Arc::from(name.as_str()), alias.clone()));
    let body = fuse_recursive(body, ctx);

    // Try to fuse: when `init` is a Patch and `body` is — or contains as
    // its leaf — a Patch (or chain-write shape) targeting the same
    // canonical root (after alias resolution). Pull those ops into the
    // init list; replace the inner write with a reference to the alias.
    let merged = try_let_init_body_fusion(&name, &alias, &init, &body, ctx);

    ctx.aliases.pop();

    match merged {
        Some(new_let) => new_let,
        None => Expr::Let {
            name,
            init: Box::new(init),
            body: Box::new(body),
        },
    }
}

/// If `init` is a Patch, return the root expression that names the
/// document being patched (so canonicalisation flows through let-aliases
/// correctly). Otherwise return the init expression unchanged.
fn init_target(init: &Expr) -> Expr {
    match init {
        Expr::Patch { root, .. } => (**root).clone(),
        other => other.clone(),
    }
}

/// Implement P3. Returns `Some(rewritten_let)` on success, leaving
/// fallback to the caller.
fn try_let_init_body_fusion(
    name: &str,
    alias: &RootRef,
    init: &Expr,
    body: &Expr,
    ctx: &FuseCtx,
) -> Option<Expr> {
    let (init_root, init_ops) = match init {
        Expr::Patch { root, ops } => (root, ops),
        _ => return None,
    };
    if init_ops.iter().any(|o| o.cond.is_some()) {
        return None;
    }
    let init_root_ref = ctx.canonical_root(init_root);
    if matches!(init_root_ref, RootRef::Composite(_)) {
        return None;
    }

    // Try the body as a direct Patch first; if not, try lifting a
    // chain-write (e.g. `x.b.set(2)`).
    let body_patch_owned;
    let body_patch: &Expr = if matches!(body, Expr::Patch { .. }) {
        body
    } else if let Some(lifted) = try_lift_chain_write(body) {
        body_patch_owned = lifted;
        &body_patch_owned
    } else {
        return None;
    };

    let (body_root, body_ops) = match body_patch {
        Expr::Patch { root, ops } => (root, ops),
        _ => return None,
    };
    if body_ops.iter().any(|o| o.cond.is_some()) {
        return None;
    }

    // Canonicalise body's root with `name` aliased to `init_root_ref`.
    let body_root_ref = canonical_root_with_alias(body_root, name, alias, ctx);
    if body_root_ref != init_root_ref {
        return None;
    }

    let mut merged = init_ops.clone();
    merged.extend(body_ops.iter().cloned());

    // The let body becomes a bare reference to `name` (which is the
    // patched doc). Equivalently we could collapse to the init, but
    // keeping the let preserves the binding's scope for any outer
    // expression that might capture it (compositional safety).
    Some(Expr::Let {
        name: name.to_string(),
        init: Box::new(Expr::Patch {
            root: init_root.clone(),
            ops: merged,
        }),
        body: Box::new(Expr::Ident(name.to_string())),
    })
}

/// Resolve `expr` as a canonical root with `name` temporarily aliased to
/// `alias`. Used in P3 because the body sees the let-bound name and we
/// need to canonicalise *as if* we were inside the body's lexical scope.
fn canonical_root_with_alias(
    expr: &Expr,
    name: &str,
    alias: &RootRef,
    ctx: &FuseCtx,
) -> RootRef {
    match expr {
        Expr::Ident(n) if n == name => alias.clone(),
        Expr::Root => RootRef::Root,
        Expr::Current => RootRef::Current(ctx.current_scope()),
        other => ctx.canonical_root(other),
    }
}

// --- Phase A analyser hook for object-field reads -------------------------

impl EffectAnalyzer {
    /// Public-in-crate facade so Phase B's object lifter can compute a
    /// summary for one field without re-implementing the dispatch.
    pub(crate) fn analyze_obj_field(&mut self, f: &ObjField) -> EffectSummary {
        self.visit_obj_field(f)
    }
}

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

    // -------- Phase C: cycle / depth safety on alias chains -----------

    #[test]
    fn alias_chain_cycle_bails_to_composite() {
        // Synthesise an alias table with a self-cycle a -> b -> a, then
        // canonicalise `a`. Without the cycle guard we'd loop forever;
        // with it we land on Composite.
        let mut ctx = FuseCtx::default();
        ctx.aliases
            .push((Arc::from("a"), RootRef::Local(Arc::from("b"))));
        ctx.aliases
            .push((Arc::from("b"), RootRef::Local(Arc::from("a"))));
        let r = ctx.canonical_root(&Expr::Ident("a".to_string()));
        assert!(
            matches!(r, RootRef::Composite(_)),
            "cycle should fall back to Composite, got {:?}",
            r
        );
    }

    #[test]
    fn alias_chain_depth_cap_bails_to_composite() {
        // Build a chain longer than `MAX_ALIAS_CHAIN_DEPTH`. Each name
        // resolves to the next; the last one resolves to nothing
        // (`Local` terminal). The walker should hit the cap and bail.
        let mut ctx = FuseCtx::default();
        let len = MAX_ALIAS_CHAIN_DEPTH + 2;
        for i in 0..len {
            let name: Arc<str> = Arc::from(format!("n{}", i));
            let next: Arc<str> = Arc::from(format!("n{}", i + 1));
            ctx.aliases.push((name, RootRef::Local(next)));
        }
        let r = ctx.canonical_root(&Expr::Ident("n0".to_string()));
        assert!(
            matches!(r, RootRef::Composite(_)),
            "over-deep chain should bail to Composite, got {:?}",
            r
        );
    }

    #[test]
    fn alias_chain_within_cap_resolves_to_root() {
        // A chain *exactly* at the cap minus one (so the walker has
        // budget to reach the bottom) bottoms out at Root.
        let mut ctx = FuseCtx::default();
        let len = MAX_ALIAS_CHAIN_DEPTH - 1;
        for i in 0..len {
            let name: Arc<str> = Arc::from(format!("n{}", i));
            let next: Arc<str> = Arc::from(format!("n{}", i + 1));
            ctx.aliases.push((name, RootRef::Local(next)));
        }
        // Final hop bottoms out at Root.
        ctx.aliases.push((
            Arc::from(format!("n{}", len)),
            RootRef::Root,
        ));
        let r = ctx.canonical_root(&Expr::Ident("n0".to_string()));
        assert_eq!(r, RootRef::Root);
    }

    #[test]
    fn pending_batch_add_and_flush_emits_let_wrapper() {
        // Phase C: adding ops to pending and flushing wraps the body
        // in `let __patch_fuse_0 = patch <root> { … } in body`.
        let mut ctx = FuseCtx::default();
        let op = PatchOp {
            path: vec![PathStep::Field("k".into())],
            val: Expr::Int(1),
            cond: None,
        };
        let _binding = ctx.add_to_batch(RootRef::Root, Expr::Root, vec![op]);
        let body = Expr::Bool(true);
        let wrapped = ctx.flush_all(body);
        match wrapped {
            Expr::Let { name, init, body } => {
                assert!(name.starts_with("__patch_fuse_"));
                assert!(matches!(*init, Expr::Patch { .. }));
                assert!(matches!(*body, Expr::Bool(true)));
            }
            other => panic!("expected Let wrapper, got {:?}", other),
        }
        // Pending should be drained.
        assert!(ctx.pending.is_empty());
    }

    #[test]
    fn flush_root_only_drains_named_root() {
        // Two pending batches against distinct roots; flush_root drains
        // only the targeted one and leaves the other intact.
        let mut ctx = FuseCtx::default();
        let op_a = PatchOp {
            path: vec![PathStep::Field("a".into())],
            val: Expr::Int(1),
            cond: None,
        };
        let op_b = PatchOp {
            path: vec![PathStep::Field("b".into())],
            val: Expr::Int(2),
            cond: None,
        };
        ctx.add_to_batch(RootRef::Root, Expr::Root, vec![op_a]);
        ctx.add_to_batch(
            RootRef::Local(Arc::from("x")),
            Expr::Ident("x".into()),
            vec![op_b],
        );
        let body = ctx.flush_root(&RootRef::Root, Expr::Bool(true));
        // Wrapper for Root should be emitted; Local("x") still pending.
        assert!(matches!(body, Expr::Let { .. }));
        assert_eq!(ctx.pending.len(), 1);
        assert!(ctx
            .pending
            .contains_key(&RootRef::Local(Arc::from("x"))));
    }
}
