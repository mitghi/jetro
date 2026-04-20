//! Logical plan IR.
//!
//! Relational-style representation of a query: `Scan`, `Filter`, `Project`,
//! `Aggregate`, `Sort`, `Limit`, `Join`.  Built from a compiled `Program`
//! by walking method calls on arrays; unrecognised opcode sequences fall
//! through as an opaque `Raw` node.
//!
//! The logical plan enables rewrites that are hard to express at the
//! opcode level: e.g. predicate pushdown, filter-then-project reorder,
//! join detection across `let` bindings.
//!
//! This is a scaffold — the rewrite rules library is intentionally small.

use std::sync::Arc;
use super::vm::{Program, Opcode, BuiltinMethod};

#[derive(Debug, Clone)]
pub enum LogicalPlan {
    /// Root scan — produces the input document.
    Scan,
    /// Navigate into a field chain from the scan.
    Path(Vec<Arc<str>>),
    /// Filter rows by a boolean predicate program.
    Filter { input: Box<LogicalPlan>, pred: Arc<Program> },
    /// Project / transform each row.
    Project { input: Box<LogicalPlan>, map: Arc<Program> },
    /// Aggregate to a single scalar.
    Aggregate { input: Box<LogicalPlan>, op: AggOp, arg: Option<Arc<Program>> },
    /// Sort rows.
    Sort { input: Box<LogicalPlan>, key: Option<Arc<Program>>, desc: bool },
    /// Limit / TopN.
    Limit { input: Box<LogicalPlan>, n: usize },
    /// Join two plans on matching keys (stubbed — detected but not yet
    /// rewritten into a fused execution).
    Join { left: Box<LogicalPlan>, right: Box<LogicalPlan>, on: Arc<Program> },
    /// Opaque fallback: opcode sequence not yet lifted.
    Raw(Arc<Program>),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AggOp { Count, Sum, Avg, Min, Max, First, Last }

impl LogicalPlan {
    /// Lift a compiled `Program` into a `LogicalPlan`.  Best-effort: falls
    /// back to `Raw` for opcode sequences that don't fit the relational shape.
    pub fn lift(program: &Program) -> LogicalPlan {
        let ops = &program.ops;
        if ops.is_empty() { return LogicalPlan::Raw(Arc::new(program.clone())); }

        let mut plan: Option<LogicalPlan> = None;
        for op in ops.iter() {
            plan = Some(match (plan.take(), op) {
                (None, Opcode::PushRoot) => LogicalPlan::Scan,
                (None, Opcode::RootChain(chain)) => {
                    LogicalPlan::Path(chain.iter().cloned().collect())
                }
                (Some(p), Opcode::GetField(k)) => match p {
                    LogicalPlan::Scan => LogicalPlan::Path(vec![k.clone()]),
                    LogicalPlan::Path(mut v) => { v.push(k.clone()); LogicalPlan::Path(v) }
                    other => LogicalPlan::Project {
                        input: Box::new(other),
                        map: Arc::new(Program {
                            ops: Arc::from(vec![Opcode::PushCurrent, Opcode::GetField(k.clone())]),
                            source: Arc::from(""), id: 0, is_structural: true,
                        }),
                    },
                }
                (Some(p), Opcode::RootChain(chain)) => {
                    let mut v: Vec<Arc<str>> = match p { LogicalPlan::Scan => vec![], _ => return LogicalPlan::Raw(Arc::new(program.clone())) };
                    for k in chain.iter() { v.push(k.clone()); }
                    LogicalPlan::Path(v)
                }
                (Some(p), Opcode::CallMethod(c)) => lift_method(p, c),
                (Some(p), Opcode::FilterMap { pred, map }) => LogicalPlan::Project {
                    input: Box::new(LogicalPlan::Filter {
                        input: Box::new(p),
                        pred:  Arc::clone(pred),
                    }),
                    map: Arc::clone(map),
                },
                (Some(p), Opcode::FilterCount(pred)) => LogicalPlan::Aggregate {
                    input: Box::new(LogicalPlan::Filter {
                        input: Box::new(p),
                        pred:  Arc::clone(pred),
                    }),
                    op:    AggOp::Count,
                    arg:   None,
                },
                (Some(p), Opcode::MapSum(f)) => LogicalPlan::Aggregate {
                    input: Box::new(p),
                    op:    AggOp::Sum,
                    arg:   Some(Arc::clone(f)),
                },
                (Some(p), Opcode::MapAvg(f)) => LogicalPlan::Aggregate {
                    input: Box::new(p),
                    op:    AggOp::Avg,
                    arg:   Some(Arc::clone(f)),
                },
                (Some(p), Opcode::TopN { n, asc }) => LogicalPlan::Limit {
                    input: Box::new(LogicalPlan::Sort {
                        input: Box::new(p),
                        key:   None,
                        desc:  !asc,
                    }),
                    n: *n,
                },
                _ => return LogicalPlan::Raw(Arc::new(program.clone())),
            });
        }
        plan.unwrap_or(LogicalPlan::Raw(Arc::new(program.clone())))
    }

    /// True if this plan is a pure aggregate (reduces to scalar).
    pub fn is_aggregate(&self) -> bool {
        matches!(self, LogicalPlan::Aggregate { .. })
    }

    /// Depth of the plan tree (for cost).
    pub fn depth(&self) -> usize {
        match self {
            LogicalPlan::Scan | LogicalPlan::Path(_) | LogicalPlan::Raw(_) => 1,
            LogicalPlan::Filter { input, .. } | LogicalPlan::Project { input, .. }
                | LogicalPlan::Sort { input, .. } | LogicalPlan::Limit { input, .. }
                | LogicalPlan::Aggregate { input, .. } => 1 + input.depth(),
            LogicalPlan::Join { left, right, .. } =>
                1 + left.depth().max(right.depth()),
        }
    }
}

fn lift_method(input: LogicalPlan, c: &Arc<super::vm::CompiledCall>) -> LogicalPlan {
    match c.method {
        BuiltinMethod::Filter if !c.sub_progs.is_empty() => LogicalPlan::Filter {
            input: Box::new(input),
            pred:  Arc::clone(&c.sub_progs[0]),
        },
        BuiltinMethod::Map if !c.sub_progs.is_empty() => LogicalPlan::Project {
            input: Box::new(input),
            map:   Arc::clone(&c.sub_progs[0]),
        },
        BuiltinMethod::Sort => LogicalPlan::Sort {
            input: Box::new(input),
            key:   c.sub_progs.first().map(Arc::clone),
            desc:  false,
        },
        BuiltinMethod::Count | BuiltinMethod::Len =>
            LogicalPlan::Aggregate { input: Box::new(input), op: AggOp::Count, arg: None },
        BuiltinMethod::Sum =>
            LogicalPlan::Aggregate { input: Box::new(input), op: AggOp::Sum, arg: c.sub_progs.first().map(Arc::clone) },
        BuiltinMethod::Avg =>
            LogicalPlan::Aggregate { input: Box::new(input), op: AggOp::Avg, arg: c.sub_progs.first().map(Arc::clone) },
        BuiltinMethod::Min =>
            LogicalPlan::Aggregate { input: Box::new(input), op: AggOp::Min, arg: None },
        BuiltinMethod::Max =>
            LogicalPlan::Aggregate { input: Box::new(input), op: AggOp::Max, arg: None },
        BuiltinMethod::First =>
            LogicalPlan::Aggregate { input: Box::new(input), op: AggOp::First, arg: None },
        BuiltinMethod::Last =>
            LogicalPlan::Aggregate { input: Box::new(input), op: AggOp::Last, arg: None },
        _ => LogicalPlan::Raw(Arc::new(Program {
            ops: Arc::from(vec![Opcode::CallMethod(Arc::clone(c))]),
            source: Arc::from(""), id: 0, is_structural: false,
        })),
    }
}

// ── Rewrite rules ─────────────────────────────────────────────────────────────

/// Push a filter down through a project when the project's map is a pure
/// field-access (equi-projection).  Enables evaluating predicate on the
/// larger pre-project rowset, which is often cheaper.
///
/// This is a skeleton — only the trivially-safe case is rewritten.
pub fn pushdown_filter(plan: LogicalPlan) -> LogicalPlan {
    match plan {
        LogicalPlan::Filter { input, pred } => match *input {
            // filter(sort(x)) → sort(filter(x))   [sort is order-preserving filter-wise]
            LogicalPlan::Sort { input: inner, key, desc } => LogicalPlan::Sort {
                input: Box::new(pushdown_filter(LogicalPlan::Filter { input: inner, pred })),
                key, desc,
            },
            other => LogicalPlan::Filter { input: Box::new(pushdown_filter(other)), pred },
        },
        LogicalPlan::Project { input, map } => LogicalPlan::Project {
            input: Box::new(pushdown_filter(*input)),
            map,
        },
        LogicalPlan::Sort { input, key, desc } => LogicalPlan::Sort {
            input: Box::new(pushdown_filter(*input)),
            key, desc,
        },
        LogicalPlan::Limit { input, n } => LogicalPlan::Limit {
            input: Box::new(pushdown_filter(*input)),
            n,
        },
        LogicalPlan::Aggregate { input, op, arg } => LogicalPlan::Aggregate {
            input: Box::new(pushdown_filter(*input)),
            op, arg,
        },
        other => other,
    }
}

// ── Lowering: LogicalPlan → Program ──────────────────────────────────────────

/// Compile a `LogicalPlan` back to a flat `Program`.  Inverse of `lift`.
/// Lifting then lowering should produce a semantically-equivalent program
/// (not necessarily byte-identical — the lowered form is canonicalised).
pub fn lower(plan: &LogicalPlan) -> Arc<Program> {
    let mut ops = Vec::new();
    emit(plan, &mut ops);
    Arc::new(Program {
        ops:           ops.into(),
        source:        Arc::from("<lowered>"),
        id:            0,
        is_structural: false,
    })
}

fn emit(plan: &LogicalPlan, ops: &mut Vec<super::vm::Opcode>) {
    use super::vm::Opcode;
    match plan {
        LogicalPlan::Scan => ops.push(Opcode::PushRoot),
        LogicalPlan::Path(ks) => {
            ops.push(Opcode::RootChain(ks.iter().cloned().collect::<Vec<_>>().into()));
        }
        LogicalPlan::Filter { input, pred } => {
            emit(input, ops);
            ops.push(Opcode::InlineFilter(Arc::clone(pred)));
        }
        LogicalPlan::Project { input, map } => {
            emit(input, ops);
            ops.push(map_as_call(map));
        }
        LogicalPlan::Sort { input, key, desc } => {
            emit(input, ops);
            ops.push(sort_as_call(key.as_ref()));
            if *desc { ops.push(reverse_call()); }
        }
        LogicalPlan::Limit { input, n } => {
            emit(input, ops);
            ops.push(Opcode::TopN { n: *n, asc: true });
        }
        LogicalPlan::Aggregate { input, op, arg } => {
            emit(input, ops);
            match op {
                AggOp::Count => ops.push(noarg_call(super::vm::BuiltinMethod::Count, "count")),
                AggOp::Sum if arg.is_some() => ops.push(Opcode::MapSum(Arc::clone(arg.as_ref().unwrap()))),
                AggOp::Avg if arg.is_some() => ops.push(Opcode::MapAvg(Arc::clone(arg.as_ref().unwrap()))),
                AggOp::Sum => ops.push(noarg_call(super::vm::BuiltinMethod::Sum, "sum")),
                AggOp::Avg => ops.push(noarg_call(super::vm::BuiltinMethod::Avg, "avg")),
                AggOp::Min => ops.push(noarg_call(super::vm::BuiltinMethod::Min, "min")),
                AggOp::Max => ops.push(noarg_call(super::vm::BuiltinMethod::Max, "max")),
                AggOp::First => ops.push(noarg_call(super::vm::BuiltinMethod::First, "first")),
                AggOp::Last  => ops.push(noarg_call(super::vm::BuiltinMethod::Last, "last")),
            }
        }
        LogicalPlan::Join { left, right: _, on: _ } => {
            // Placeholder: emit left only (no fused join runtime yet).
            emit(left, ops);
        }
        LogicalPlan::Raw(p) => {
            for op in p.ops.iter() { ops.push(op.clone()); }
        }
    }
}

fn noarg_call(method: super::vm::BuiltinMethod, name: &str) -> super::vm::Opcode {
    use super::vm::{Opcode, CompiledCall};
    Opcode::CallMethod(Arc::new(CompiledCall {
        method, name: Arc::from(name),
        sub_progs: Arc::from(Vec::new()),
        orig_args: Arc::from(Vec::new()),
    }))
}

fn reverse_call() -> super::vm::Opcode {
    noarg_call(super::vm::BuiltinMethod::Reverse, "reverse")
}

fn map_as_call(map: &Arc<Program>) -> super::vm::Opcode {
    use super::vm::{Opcode, CompiledCall, BuiltinMethod};
    Opcode::CallMethod(Arc::new(CompiledCall {
        method:    BuiltinMethod::Map,
        name:      Arc::from("map"),
        sub_progs: Arc::from(vec![Arc::clone(map)]),
        orig_args: Arc::from(Vec::new()),
    }))
}

fn sort_as_call(key: Option<&Arc<Program>>) -> super::vm::Opcode {
    use super::vm::{Opcode, CompiledCall, BuiltinMethod};
    let sub_progs: Vec<Arc<Program>> = key.map(|k| vec![Arc::clone(k)]).unwrap_or_default();
    Opcode::CallMethod(Arc::new(CompiledCall {
        method:    BuiltinMethod::Sort,
        name:      Arc::from("sort"),
        sub_progs: sub_progs.into(),
        orig_args: Arc::from(Vec::new()),
    }))
}

// ── Join detection ────────────────────────────────────────────────────────────

/// Walk a `LogicalPlan` looking for filter predicates that compare two
/// distinct identifiers — a candidate equi-join between correlated scans.
/// Returns a vector of (left-path, right-path) fragment pairs per detected
/// candidate.  Wiring actual `Join` rewrite is future work.
pub fn detect_join_candidates(plan: &LogicalPlan) -> Vec<JoinCandidate> {
    let mut out = Vec::new();
    walk(plan, &mut out);
    out
}

#[derive(Debug, Clone)]
pub struct JoinCandidate {
    /// Textual description of left side (from pred source or ident).
    pub left:  String,
    pub right: String,
}

fn walk(plan: &LogicalPlan, out: &mut Vec<JoinCandidate>) {
    match plan {
        LogicalPlan::Filter { input, pred } => {
            if let Some(j) = detect_eq_join(pred) { out.push(j); }
            walk(input, out);
        }
        LogicalPlan::Project { input, .. }
            | LogicalPlan::Sort { input, .. }
            | LogicalPlan::Limit { input, .. }
            | LogicalPlan::Aggregate { input, .. } => walk(input, out),
        LogicalPlan::Join { left, right, .. } => { walk(left, out); walk(right, out); }
        _ => {}
    }
}

fn detect_eq_join(pred: &Arc<Program>) -> Option<JoinCandidate> {
    use crate::v2::vm::Opcode;
    // Look for pattern: LoadIdent + GetField* + LoadIdent + GetField* + Eq
    let ops = &pred.ops;
    if ops.len() < 3 { return None; }
    let last = ops.last()?;
    if !matches!(last, Opcode::Eq) { return None; }
    // Split ops by finding the second LoadIdent (crude): two independent
    // chains, both starting with a LoadIdent.
    let mut ident_positions: Vec<usize> = ops.iter().enumerate()
        .filter(|(_, o)| matches!(o, Opcode::LoadIdent(_)))
        .map(|(i, _)| i).collect();
    if ident_positions.len() != 2 { return None; }
    let a = describe_chain(&ops[ident_positions[0]..ident_positions[1]]);
    let b = describe_chain(&ops[ident_positions[1]..ops.len()-1]);
    if a == b { return None; }
    Some(JoinCandidate { left: a, right: b })
}

fn describe_chain(ops: &[super::vm::Opcode]) -> String {
    use super::vm::Opcode;
    let mut s = String::new();
    for op in ops {
        match op {
            Opcode::LoadIdent(n) => s.push_str(n),
            Opcode::GetField(k)  => { s.push('.'); s.push_str(k); }
            _ => {}
        }
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::v2::vm::Compiler;

    #[test]
    fn lift_path() {
        let p = Compiler::compile_str("$.store.books").unwrap();
        match LogicalPlan::lift(&p) {
            LogicalPlan::Path(v) => {
                assert_eq!(v.len(), 2);
                assert_eq!(v[0].as_ref(), "store");
                assert_eq!(v[1].as_ref(), "books");
            }
            other => panic!("expected Path, got {:?}", other),
        }
    }

    #[test]
    fn lift_filter_map() {
        let p = Compiler::compile_str("$.books.filter(@.price > 10).map(@.title)").unwrap();
        let plan = LogicalPlan::lift(&p);
        // filter+map fuses to FilterMap opcode → lifts to Project(Filter(..))
        assert!(plan.depth() >= 2);
    }

    #[test]
    fn lift_aggregate() {
        let p = Compiler::compile_str("$.books.count()").unwrap();
        let plan = LogicalPlan::lift(&p);
        assert!(plan.is_aggregate());
    }

    #[test]
    fn roundtrip_lower_preserves_semantics() {
        use crate::v2::vm::VM;
        use serde_json::json;
        let doc = json!({"store": {"books": [{"price": 20}, {"price": 5}]}});
        let p = Compiler::compile_str("$.store.books.filter(@.price > 10).count()").unwrap();
        let plan = LogicalPlan::lift(&p);
        let lowered = lower(&plan);
        let mut vm = VM::new();
        let original = vm.execute(&p, &doc).unwrap();
        let round    = vm.execute(&lowered, &doc).unwrap();
        assert_eq!(original, round);
    }

    #[test]
    fn detect_join_candidates_finds_equi_join() {
        // Two different identifiers compared by eq → join candidate.
        let p = Compiler::compile_str("$.x.filter(a.id == b.id)").unwrap();
        let plan = LogicalPlan::lift(&p);
        let candidates = detect_join_candidates(&plan);
        assert!(!candidates.is_empty(), "should detect a.id == b.id as join");
    }
}
