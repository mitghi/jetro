//! Pipeline IR — pull-based query plan that replaces hand-written
//! peephole fusions in `vm.rs`.
//!
//! Goal: a query like `$.orders.filter(total > 100).map(id).count()`
//! lowers to:
//!
//! ```ignore
//! Pipeline {
//!     source: Source::Field { base: Source::Root, key: "orders" },
//!     stages: vec![
//!         Stage::Filter(<prog: total > 100>),
//!         Stage::Map(<prog: id>),
//!     ],
//!     sink: Sink::Count,
//! }
//! ```
//!
//! Execution = single outer loop in [`Sink::run`] that pulls one
//! element from the source, threads it through the stages, writes
//! into the sink — no `Vec<Val>` between stages.
//!
//! Phase 1 (this module): pull-based [`Pipeline`] / [`Stage`] /
//! [`Source`] / [`Sink`] + a lowering path that handles a small
//! initial shape set (`Field`-chain source, `Filter`/`Map`/`Take`/
//! `Skip` stages, `Count`/`Sum`/`Collect` sinks).  Anything outside
//! the supported shape set falls back to the existing opcode path
//! by returning `None` from [`Pipeline::lower`].
//!
//! Phase 2 will add rewrite rules; Phase 3 will swap the per-element
//! `pull_next` for a per-batch `pull_batch` over `IntVec`/`FloatVec`/
//! `StrVec` columnar lanes.
//!
//! See `memory/project_pipeline_ir.md` for the full plan.

use std::sync::Arc;

use crate::ast::Expr;
use crate::eval::value::Val;
use crate::eval::EvalError;

// ── Plan types ───────────────────────────────────────────────────────────────

/// Where a pipeline starts.  Currently a small set; Phase 2/3 add
/// `DeepScan(key)` (tape byte-scan), `Range(i64, i64)`, and
/// `Tape(Arc<TapeData>)`.
#[derive(Debug, Clone)]
pub enum Source {
    /// Pull from a concrete `Val::Arr` / `Val::IntVec` / `Val::FloatVec` /
    /// `Val::StrVec` / `Val::ObjVec` already on the stack.
    Receiver(Val),
    /// Walk `$.<keys[0]>.<keys[1]>…` from the document root, then iterate
    /// the array at the end of the chain.  `keys.is_empty()` means the
    /// document root itself is the iterable.
    FieldChain { keys: Arc<[Arc<str>]> },
}

/// A pull-based stage.  Filter / Map carry a pre-compiled `Program`
/// that runs against each row's `@` (current item) bound as the VM
/// root.  Programs are compiled once at lowering time and reused per
/// row — drops the per-row tree-walker dispatch cost that the initial
/// substrate paid.
#[derive(Debug, Clone)]
pub enum Stage {
    /// `.filter(pred)` — drops elements where `pred` is falsy.
    Filter(Arc<crate::vm::Program>),
    /// `.map(f)` — replaces each element with `f(@)`.
    Map(Arc<crate::vm::Program>),
    /// `.take(n)` — yields at most `n` elements, then completes.
    Take(usize),
    /// `.skip(n)` — drops the first `n` elements.
    Skip(usize),
}

/// Where pipeline output lands.  Determines the result type.
///
/// The fused variants (`SumMap`, `CountIf`, `SumFilterMap`,
/// `CountFilter`) are produced by Phase 2 rewrite rules during
/// lowering — they collapse adjacent `Map`/`Filter` stages and the
/// sink into a single inner-loop kernel, eliminating the
/// per-row exec call per stage.
#[derive(Debug, Clone)]
pub enum Sink {
    /// Materialise every element into a `Val::Arr`.
    Collect,
    /// `.count()` / `.len()` — yield the number of elements that
    /// reached the sink as a `Val::Int`.
    Count,
    /// `.sum()` over numerics — yields `Val::Int` or `Val::Float`.
    Sum,
    /// `Map(prog) ∘ Sum`.  Inner loop computes `prog(@)` and
    /// accumulates numerically; never materialises a Vec.
    SumMap(Arc<crate::vm::Program>),
    /// `Filter(prog) ∘ Count`.  Inner loop runs `prog(@)`; counts
    /// truthy.
    CountIf(Arc<crate::vm::Program>),
    /// `Filter(pred) ∘ Map(f) ∘ Sum`.
    SumFilterMap(Arc<crate::vm::Program>, Arc<crate::vm::Program>),
}

#[derive(Debug, Clone)]
pub struct Pipeline {
    pub source: Source,
    pub stages: Vec<Stage>,
    pub sink:   Sink,
}

// ── Lowering ─────────────────────────────────────────────────────────────────

impl Pipeline {
    /// Try to lower an `Expr` into a Pipeline.  Returns `None` for any
    /// shape this Phase 1 substrate doesn't yet handle — caller falls
    /// back to the existing opcode compilation path.
    ///
    /// Supported (Phase 1):
    ///   - `$.k1.k2…kN.<stage>*.<sink>` where each `kN` is a plain Field,
    ///     stages are zero-or-more of `filter` / `map` / `take` / `skip`,
    ///     and the sink is `count` / `len` / `sum` / nothing (Collect).
    ///
    /// Not yet supported and returns `None`:
    ///   - Any non-Root base
    ///   - Any non-Field step before the first method (e.g. `[idx]`)
    ///   - Lambda methods (`map(@.x + 1)` is fine; `map(lambda x: …)` is not)
    ///   - Any unrecognised method in stage position
    pub fn lower(expr: &Expr) -> Option<Pipeline> {
        use crate::ast::{Step, Arg};
        let (base, steps) = match expr {
            Expr::Chain(b, s) => (b.as_ref(), s.as_slice()),
            _ => return None,
        };
        if !matches!(base, Expr::Root) { return None; }

        // Find where the field-chain prefix ends and stages begin.
        let mut field_end = 0;
        for s in steps {
            match s {
                Step::Field(_) => field_end += 1,
                _ => break,
            }
        }
        // Phase 1 deliberately does not lower bare `$.<method>` shapes
        // (no field-chain prefix) because the existing fused opcodes
        // (MapSplitLenSum, FilterFieldCmpLitMapField, etc.) often beat
        // a generic pull-based pipeline for those.  Field-chain prefix
        // signals a "scan over a sub-array" intent — the pipeline's
        // sweet spot.
        if field_end == 0 { return None; }

        let keys: Arc<[Arc<str>]> = steps[..field_end].iter()
            .map(|s| match s { Step::Field(k) => Arc::<str>::from(k.as_str()), _ => unreachable!() })
            .collect::<Vec<_>>().into();

        // Decode the trailing methods into stages + a sink.
        // Compile each filter / map sub-Expr to a Program once so
        // Pipeline::run can reuse it per row.  Sub-programs run against
        // the current item bound as the VM's root, so `@.field` and
        // `@` references resolve to the row.
        let mut stages: Vec<Stage> = Vec::new();
        let mut sink: Sink = Sink::Collect;
        let trailing = &steps[field_end..];
        for (i, s) in trailing.iter().enumerate() {
            let is_last = i == trailing.len() - 1;
            match s {
                Step::Method(name, args) => {
                    match (name.as_str(), args.len(), is_last) {
                        ("filter", 1, _) => {
                            let prog = compile_subexpr(&args[0])?;
                            stages.push(Stage::Filter(prog));
                        }
                        ("map", 1, _) => {
                            let prog = compile_subexpr(&args[0])?;
                            stages.push(Stage::Map(prog));
                        }
                        ("take", 1, _) => {
                            let n = match &args[0] {
                                Arg::Pos(Expr::Int(n)) if *n >= 0 => *n as usize,
                                _ => return None,
                            };
                            stages.push(Stage::Take(n));
                        }
                        ("skip", 1, _) => {
                            let n = match &args[0] {
                                Arg::Pos(Expr::Int(n)) if *n >= 0 => *n as usize,
                                _ => return None,
                            };
                            stages.push(Stage::Skip(n));
                        }
                        ("count", 0, true) | ("len", 0, true) => sink = Sink::Count,
                        ("sum", 0, true) => sink = Sink::Sum,
                        _ => return None,
                    }
                }
                _ => return None,
            }
        }

        let mut p = Pipeline { source: Source::FieldChain { keys }, stages, sink };
        rewrite(&mut p);
        Some(p)
    }
}

/// Apply algebraic rewrite rules until fixed point or a fuel limit
/// expires.  Each rule shrinks the stage vector by collapsing into a
/// fused sink — strictly monotonic, so no cycle risk.
fn rewrite(p: &mut Pipeline) {
    let mut fuel = 8usize;
    while fuel > 0 {
        fuel -= 1;
        let last_two = if p.stages.len() >= 2 {
            Some((p.stages.len() - 2, p.stages.len() - 1))
        } else { None };

        // Rule: Filter(pred) ∘ Map(f) ∘ Sum  →  SumFilterMap(pred, f)
        if let (Some((i_pred, i_map)), Sink::Sum) = (last_two, &p.sink) {
            if let (Stage::Filter(pred), Stage::Map(map)) =
                (&p.stages[i_pred], &p.stages[i_map])
            {
                let pred = Arc::clone(pred);
                let map  = Arc::clone(map);
                p.stages.truncate(i_pred);
                p.sink = Sink::SumFilterMap(pred, map);
                continue;
            }
        }

        // Rule: Map(f) ∘ Sum → SumMap(f)
        if let (Some(last), Sink::Sum) = (p.stages.last(), &p.sink) {
            if let Stage::Map(prog) = last {
                let prog = Arc::clone(prog);
                p.stages.pop();
                p.sink = Sink::SumMap(prog);
                continue;
            }
        }

        // Rule: Filter(p) ∘ Count → CountIf(p)
        if let (Some(last), Sink::Count) = (p.stages.last(), &p.sink) {
            if let Stage::Filter(prog) = last {
                let prog = Arc::clone(prog);
                p.stages.pop();
                p.sink = Sink::CountIf(prog);
                continue;
            }
        }

        break; // no rule matched this round
    }
}

// ── Execution ────────────────────────────────────────────────────────────────

impl Pipeline {
    /// Execute the pipeline against `root`, returning the sink's
    /// produced [`Val`].
    pub fn run(&self, root: &Val) -> Result<Val, EvalError> {
        // One VM owned by the pull loop — shared across stage program
        // calls so VM compile / path caches amortise across the row
        // sweep.  Constructing a fresh VM per row regresses 250x.
        let mut vm = crate::vm::VM::new();

        // Resolve source to an iterable Val::Arr-like sequence.
        let recv = match &self.source {
            Source::Receiver(v) => v.clone(),
            Source::FieldChain { keys } => walk_field_chain(root, keys),
        };

        // Pull-based stage chain.  At Phase 1 the inner loop materialises
        // elements one at a time as `Val`; Phase 3 will switch this to a
        // per-batch pull over columnar lanes.
        let mut taken: usize = 0;
        let mut skipped: usize = 0;

        let iter: Box<dyn Iterator<Item = Val>> = match &recv {
            Val::Arr(a)      => Box::new(a.as_ref().clone().into_iter()),
            Val::IntVec(a)   => Box::new(a.iter().map(|n| Val::Int(*n)).collect::<Vec<_>>().into_iter()),
            Val::FloatVec(a) => Box::new(a.iter().map(|f| Val::Float(*f)).collect::<Vec<_>>().into_iter()),
            Val::StrVec(a)   => Box::new(a.iter().map(|s| Val::Str(Arc::clone(s))).collect::<Vec<_>>().into_iter()),
            // Anything else (scalar, Obj, …): single-element "iterator".
            _ => Box::new(std::iter::once(recv.clone())),
        };

        // Sink accumulators.
        let mut acc_collect: Vec<Val> = Vec::new();
        let mut acc_count:   i64 = 0;
        let mut acc_sum_i:   i64 = 0;
        let mut acc_sum_f:   f64 = 0.0;
        let mut sum_floated: bool = false;

        'outer: for mut item in iter {
            // Apply stages in order.
            for stage in &self.stages {
                match stage {
                    Stage::Skip(n) => {
                        if skipped < *n { skipped += 1; continue 'outer; }
                    }
                    Stage::Take(n) => {
                        if taken >= *n { break 'outer; }
                    }
                    Stage::Filter(prog) => {
                        if !is_truthy(&apply_item_root(&mut vm, &item, prog)?) {
                            continue 'outer;
                        }
                    }
                    Stage::Map(prog) => {
                        item = apply_item_root(&mut vm, &item, prog)?;
                    }
                }
            }

            // Sink.
            match &self.sink {
                Sink::Collect => acc_collect.push(item),
                Sink::Count   => acc_count += 1,
                Sink::Sum     => match item {
                    Val::Int(n)   => if sum_floated { acc_sum_f += n as f64 } else { acc_sum_i += n },
                    Val::Float(f) => {
                        if !sum_floated { acc_sum_f = acc_sum_i as f64; sum_floated = true; }
                        acc_sum_f += f;
                    }
                    _ => {}
                },
                // Fused sinks consume the (already-stage-processed) item directly.
                Sink::SumMap(prog) => {
                    let v = apply_item_root(&mut vm, &item, prog)?;
                    sum_acc(&mut acc_sum_i, &mut acc_sum_f, &mut sum_floated, &v);
                }
                Sink::CountIf(prog) => {
                    if is_truthy(&apply_item_root(&mut vm, &item, prog)?) { acc_count += 1; }
                }
                Sink::SumFilterMap(pred, map) => {
                    if is_truthy(&apply_item_root(&mut vm, &item, pred)?) {
                        let v = apply_item_root(&mut vm, &item, map)?;
                        sum_acc(&mut acc_sum_i, &mut acc_sum_f, &mut sum_floated, &v);
                    }
                }
            }
            taken += 1;
        }

        Ok(match &self.sink {
            Sink::Collect          => Val::arr(acc_collect),
            Sink::Count            => Val::Int(acc_count),
            Sink::CountIf(_)       => Val::Int(acc_count),
            Sink::Sum
            | Sink::SumMap(_)
            | Sink::SumFilterMap(_, _) =>
                if sum_floated { Val::Float(acc_sum_f) } else { Val::Int(acc_sum_i) },
        })
    }
}

#[inline]
fn sum_acc(acc_i: &mut i64, acc_f: &mut f64, floated: &mut bool, v: &Val) {
    match v {
        Val::Int(n)   => if *floated { *acc_f += *n as f64 } else { *acc_i += *n },
        Val::Float(f) => {
            if !*floated { *acc_f = *acc_i as f64; *floated = true; }
            *acc_f += *f;
        }
        _ => {}
    }
}

/// Compile a `.filter(arg)` / `.map(arg)` sub-expression into a `Program`
/// the VM can run against a row's `@`.  The argument's expression is
/// wrapped so that bare-ident shorthand (`map(total)`) becomes
/// `@.total` for evaluation against the current row — the tree-walker
/// applies the same rule via `apply_item_mut` but we have to be
/// explicit when emitting bytecode.
fn compile_subexpr(arg: &crate::ast::Arg) -> Option<Arc<crate::vm::Program>> {
    use crate::ast::{Arg, Expr, Step};
    let inner = match arg { Arg::Pos(e) => e, _ => return None };
    let rooted: Expr = match inner {
        // Bare ident `total` → `@.total`
        Expr::Ident(name) => Expr::Chain(
            Box::new(Expr::Current),
            vec![Step::Field(name.clone())],
        ),
        // `@…` chains: keep base = Current, accept as-is
        Expr::Chain(base, _) if matches!(base.as_ref(), Expr::Current) => inner.clone(),
        // Anything else: wrap as-is — VM will resolve `@` via Current refs.
        other => other.clone(),
    };
    Some(Arc::new(crate::vm::Compiler::compile(&rooted, "")))
}

// ── Helpers ──────────────────────────────────────────────────────────────────

/// Walk `$.k1.k2…` from `root`, returning `Val::Null` on any miss.
/// Used by `Source::FieldChain` resolution.
fn walk_field_chain(root: &Val, keys: &[Arc<str>]) -> Val {
    let mut cur = root.clone();
    for k in keys {
        cur = cur.get_field(k.as_ref());
    }
    cur
}

/// Evaluate `prog` against `item` as the VM root using a long-lived
/// VM borrowed from the caller (Pipeline::run owns one per query).
/// Sharing the VM amortises its compile / path caches over the whole
/// pull loop instead of paying construction per row.
#[inline]
fn apply_item_root(vm: &mut crate::vm::VM, item: &Val, prog: &crate::vm::Program) -> Result<Val, EvalError> {
    vm.execute_val_raw(prog, item.clone())
}

#[inline]
fn is_truthy(v: &Val) -> bool {
    match v {
        Val::Null            => false,
        Val::Bool(b)         => *b,
        Val::Int(n)          => *n != 0,
        Val::Float(f)        => *f != 0.0,
        Val::Str(s)          => !s.is_empty(),
        Val::StrSlice(r)     => !r.as_str().is_empty(),
        Val::Arr(a)          => !a.is_empty(),
        Val::IntVec(a)       => !a.is_empty(),
        Val::FloatVec(a)     => !a.is_empty(),
        Val::StrVec(a)       => !a.is_empty(),
        Val::StrSliceVec(a)  => !a.is_empty(),
        Val::Obj(m)          => !m.is_empty(),
        Val::ObjSmall(p)     => !p.is_empty(),
        Val::ObjVec(d)       => !d.cells.is_empty(),
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser;

    fn lower_query(q: &str) -> Option<Pipeline> {
        let expr = parser::parse(q).ok()?;
        Pipeline::lower(&expr)
    }

    #[test]
    fn lower_field_chain_only() {
        let p = lower_query("$.a.b.c").unwrap();
        assert!(matches!(p.source, Source::FieldChain { .. }));
        assert!(p.stages.is_empty());
        assert!(matches!(p.sink, Sink::Collect));
    }

    #[test]
    fn lower_filter_map_count() {
        let p = lower_query("$.orders.filter(total > 100).map(id).count()").unwrap();
        assert_eq!(p.stages.len(), 2);
        assert!(matches!(p.stages[0], Stage::Filter(_)));
        assert!(matches!(p.stages[1], Stage::Map(_)));
        assert!(matches!(p.sink, Sink::Count));
    }

    #[test]
    fn lower_take_skip_sum() {
        let p = lower_query("$.xs.skip(2).take(5).sum()").unwrap();
        assert_eq!(p.stages.len(), 2);
        assert!(matches!(p.stages[0], Stage::Skip(2)));
        assert!(matches!(p.stages[1], Stage::Take(5)));
        assert!(matches!(p.sink, Sink::Sum));
    }

    #[test]
    fn lower_returns_none_for_unsupported_shape() {
        // Lambda-bodied filter not yet supported.
        assert!(lower_query("$.xs.group_by(status)").is_none());
        // Non-root base.
        assert!(lower_query("@.x.filter(y > 0)").is_none());
    }

    #[test]
    fn run_count_on_simple_array() {
        use serde_json::json;
        let doc: Val = (&json!({"orders":[
            {"total": 50}, {"total": 150}, {"total": 200}
        ]})).into();
        let p = lower_query("$.orders.filter(total > 100).count()").unwrap();
        let out = p.run(&doc).unwrap();
        assert_eq!(out, Val::Int(2));
    }

    #[test]
    fn run_sum_on_simple_array() {
        use serde_json::json;
        let doc: Val = (&json!({"xs":[1, 2, 3, 4, 5]})).into();
        let p = lower_query("$.xs.sum()").unwrap();
        let out = p.run(&doc).unwrap();
        assert_eq!(out, Val::Int(15));
    }

    #[test]
    fn run_filter_map_sum() {
        use serde_json::json;
        let doc: Val = (&json!({"orders":[
            {"id": 1, "total": 50},
            {"id": 2, "total": 150},
            {"id": 3, "total": 200}
        ]})).into();
        let p = lower_query("$.orders.filter(total > 100).map(total).sum()").unwrap();
        let out = p.run(&doc).unwrap();
        assert_eq!(out, Val::Int(350));
    }

    #[test]
    fn run_take_skip() {
        use serde_json::json;
        let doc: Val = (&json!({"xs":[10, 20, 30, 40, 50]})).into();
        let p = lower_query("$.xs.skip(1).take(2).sum()").unwrap();
        let out = p.run(&doc).unwrap();
        assert_eq!(out, Val::Int(50));   // 20 + 30
    }
}
