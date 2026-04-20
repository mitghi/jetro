//! Control-flow graph for v2 programs.
//!
//! v2 opcodes are linear within a program, but sub-programs (inside
//! `AndOp`, `OrOp`, `CoalesceOp`, `FilterMap`, etc.) introduce
//! conditional / looped branches.  A `BasicBlock` holds a contiguous
//! non-branching op slice; a `CFG` is a tree of blocks joined by
//! typed edges.
//!
//! This is enough structure for classic analyses (liveness, dominators)
//! to operate on — v2's restricted branch set means reducible CFGs only.

use std::sync::Arc;
use super::vm::{Program, Opcode};

/// A basic block: a straight-line opcode slice with no inner branches.
/// `branches` lists the sub-program blocks reached at the block's end.
#[derive(Debug, Clone)]
pub struct BasicBlock {
    pub id:       usize,
    pub ops:      Vec<Opcode>,
    pub branches: Vec<EdgeKind>,
}

/// Edge between blocks describing how control flows.
#[derive(Debug, Clone)]
pub enum EdgeKind {
    /// Conditional short-circuit: `AndOp` / `OrOp` — right side executed
    /// only when left is truthy / falsy respectively.
    ShortCircuit { target: usize, condition: Condition },
    /// Null-coalesce branch taken when left is null.
    Coalesce { target: usize },
    /// Per-element loop (filter / map body).
    Loop { target: usize, name: &'static str },
    /// Destructure / bind — always taken, introduces new scope.
    Bind { target: usize },
    /// Comprehension body.
    Comp { target: usize, part: CompPart },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Condition { IfTruthy, IfFalsy, IfNull }

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompPart { Expr, Iter, Cond, Key, Val }

#[derive(Debug, Clone)]
pub struct Cfg {
    pub blocks: Vec<BasicBlock>,
    pub entry:  usize,
}

impl Cfg {
    /// Build a CFG from a compiled program.  Sub-programs become their
    /// own blocks recursively; the returned `entry` is the root block id.
    pub fn build(program: &Program) -> Cfg {
        let mut cfg = Cfg { blocks: Vec::new(), entry: 0 };
        cfg.entry = build_block(&mut cfg, &program.ops);
        cfg
    }

    /// Number of basic blocks in the graph.
    pub fn size(&self) -> usize { self.blocks.len() }

    /// Reachable blocks from `entry` via BFS.
    pub fn reachable(&self) -> Vec<usize> {
        let mut visited = vec![false; self.blocks.len()];
        let mut queue = vec![self.entry];
        let mut out   = Vec::new();
        while let Some(id) = queue.pop() {
            if visited[id] { continue; }
            visited[id] = true;
            out.push(id);
            for e in &self.blocks[id].branches {
                let t = edge_target(e);
                queue.push(t);
            }
        }
        out
    }

    /// Compute immediate dominators by iterative dataflow (Cooper–Harvey–Kennedy).
    pub fn dominators(&self) -> Vec<Option<usize>> {
        let n = self.blocks.len();
        let mut doms: Vec<Option<usize>> = vec![None; n];
        if n == 0 { return doms; }
        doms[self.entry] = Some(self.entry);
        // Post-order traversal.
        let order = self.reachable();
        let mut changed = true;
        while changed {
            changed = false;
            for &b in order.iter().rev() {
                if b == self.entry { continue; }
                // Find all predecessors.
                let preds: Vec<usize> = (0..n).filter(|&p|
                    self.blocks[p].branches.iter().any(|e| edge_target(e) == b)
                ).collect();
                let mut new_idom: Option<usize> = None;
                for p in preds {
                    if doms[p].is_none() { continue; }
                    new_idom = Some(match new_idom {
                        None => p,
                        Some(cur) => intersect(&doms, p, cur),
                    });
                }
                if doms[b] != new_idom {
                    doms[b] = new_idom;
                    changed = true;
                }
            }
        }
        doms
    }
}

fn edge_target(e: &EdgeKind) -> usize {
    match e {
        EdgeKind::ShortCircuit { target, .. }
            | EdgeKind::Coalesce { target }
            | EdgeKind::Loop { target, .. }
            | EdgeKind::Bind { target }
            | EdgeKind::Comp { target, .. } => *target,
    }
}

fn intersect(doms: &[Option<usize>], mut a: usize, mut b: usize) -> usize {
    while a != b {
        while a > b { a = doms[a].unwrap_or(a); if a == doms[a].unwrap_or(a) && a != b { break; } }
        while b > a { b = doms[b].unwrap_or(b); if b == doms[b].unwrap_or(b) && a != b { break; } }
        if doms[a].map_or(true, |d| d == a) && doms[b].map_or(true, |d| d == b) { break; }
    }
    a
}

fn build_block(cfg: &mut Cfg, ops: &[Opcode]) -> usize {
    let id = cfg.blocks.len();
    cfg.blocks.push(BasicBlock { id, ops: Vec::new(), branches: Vec::new() });
    let mut straight: Vec<Opcode> = Vec::new();
    let mut branches: Vec<EdgeKind> = Vec::new();
    for op in ops {
        match op {
            Opcode::AndOp(p) => {
                let t = build_block(cfg, &p.ops);
                branches.push(EdgeKind::ShortCircuit { target: t, condition: Condition::IfTruthy });
                straight.push(op.clone());
            }
            Opcode::OrOp(p) => {
                let t = build_block(cfg, &p.ops);
                branches.push(EdgeKind::ShortCircuit { target: t, condition: Condition::IfFalsy });
                straight.push(op.clone());
            }
            Opcode::CoalesceOp(p) => {
                let t = build_block(cfg, &p.ops);
                branches.push(EdgeKind::Coalesce { target: t });
                straight.push(op.clone());
            }
            Opcode::InlineFilter(p) | Opcode::FilterCount(p)
                | Opcode::FindFirst(p) | Opcode::FindOne(p)
                | Opcode::MapSum(p) | Opcode::MapAvg(p) => {
                let t = build_block(cfg, &p.ops);
                branches.push(EdgeKind::Loop { target: t, name: "filter" });
                straight.push(op.clone());
            }
            Opcode::FilterMap { pred, map } => {
                let tp = build_block(cfg, &pred.ops);
                let tm = build_block(cfg, &map.ops);
                branches.push(EdgeKind::Loop { target: tp, name: "filter" });
                branches.push(EdgeKind::Loop { target: tm, name: "map" });
                straight.push(op.clone());
            }
            Opcode::FilterFilter { p1, p2 } => {
                let t1 = build_block(cfg, &p1.ops);
                let t2 = build_block(cfg, &p2.ops);
                branches.push(EdgeKind::Loop { target: t1, name: "filter1" });
                branches.push(EdgeKind::Loop { target: t2, name: "filter2" });
                straight.push(op.clone());
            }
            Opcode::MapMap { f1, f2 } => {
                let t1 = build_block(cfg, &f1.ops);
                let t2 = build_block(cfg, &f2.ops);
                branches.push(EdgeKind::Loop { target: t1, name: "map1" });
                branches.push(EdgeKind::Loop { target: t2, name: "map2" });
                straight.push(op.clone());
            }
            Opcode::LetExpr { body, .. } => {
                let t = build_block(cfg, &body.ops);
                branches.push(EdgeKind::Bind { target: t });
                straight.push(op.clone());
            }
            Opcode::ListComp(s) | Opcode::SetComp(s) => {
                let te = build_block(cfg, &s.expr.ops);
                let ti = build_block(cfg, &s.iter.ops);
                branches.push(EdgeKind::Comp { target: te, part: CompPart::Expr });
                branches.push(EdgeKind::Comp { target: ti, part: CompPart::Iter });
                if let Some(c) = &s.cond {
                    let tc = build_block(cfg, &c.ops);
                    branches.push(EdgeKind::Comp { target: tc, part: CompPart::Cond });
                }
                straight.push(op.clone());
            }
            Opcode::DictComp(s) => {
                let tk = build_block(cfg, &s.key.ops);
                let tv = build_block(cfg, &s.val.ops);
                let ti = build_block(cfg, &s.iter.ops);
                branches.push(EdgeKind::Comp { target: tk, part: CompPart::Key });
                branches.push(EdgeKind::Comp { target: tv, part: CompPart::Val });
                branches.push(EdgeKind::Comp { target: ti, part: CompPart::Iter });
                if let Some(c) = &s.cond {
                    let tc = build_block(cfg, &c.ops);
                    branches.push(EdgeKind::Comp { target: tc, part: CompPart::Cond });
                }
                straight.push(op.clone());
            }
            Opcode::CallMethod(c) | Opcode::CallOptMethod(c) => {
                for p in c.sub_progs.iter() {
                    let t = build_block(cfg, &p.ops);
                    branches.push(EdgeKind::Loop { target: t, name: "method" });
                }
                straight.push(op.clone());
            }
            _ => straight.push(op.clone()),
        }
    }
    cfg.blocks[id].ops = straight;
    cfg.blocks[id].branches = branches;
    id
}

// Silence unused-Arc warning.
#[allow(dead_code)]
fn _use_arc<T>(_: Arc<T>) {}

// ── Liveness analysis ────────────────────────────────────────────────────────

use std::collections::HashSet;

/// Per-block live-in / live-out sets of identifier names.
#[derive(Debug, Clone, Default)]
pub struct Liveness {
    pub live_in:  Vec<HashSet<Arc<str>>>,
    pub live_out: Vec<HashSet<Arc<str>>>,
}

impl Cfg {
    /// Compute live-in / live-out sets for identifiers (variables) across
    /// the CFG using standard backward dataflow.  A variable is live at a
    /// point if a `LoadIdent` for it exists on some path to the exit.
    pub fn liveness(&self) -> Liveness {
        let n = self.blocks.len();
        let mut live_in:  Vec<HashSet<Arc<str>>> = vec![HashSet::new(); n];
        let mut live_out: Vec<HashSet<Arc<str>>> = vec![HashSet::new(); n];

        // Per-block USE (read before any DEF) and DEF (written by LetExpr/BindVar).
        let (usev, defv) = (0..n).map(|i| compute_use_def(&self.blocks[i].ops))
            .fold((Vec::new(), Vec::new()), |(mut u, mut d), (bu, bd)| { u.push(bu); d.push(bd); (u, d) });

        let mut changed = true;
        while changed {
            changed = false;
            for b in 0..n {
                // live_out[b] = U over successors s: live_in[s]
                let mut new_out: HashSet<Arc<str>> = HashSet::new();
                for e in &self.blocks[b].branches {
                    let s = edge_target(e);
                    new_out.extend(live_in[s].iter().cloned());
                }
                // live_in[b] = use[b] ∪ (live_out[b] − def[b])
                let mut new_in = usev[b].clone();
                for v in &new_out {
                    if !defv[b].contains(v) { new_in.insert(v.clone()); }
                }
                if new_in != live_in[b]  { live_in[b]  = new_in;  changed = true; }
                if new_out != live_out[b]{ live_out[b] = new_out; changed = true; }
            }
        }
        Liveness { live_in, live_out }
    }
}

fn compute_use_def(ops: &[Opcode]) -> (HashSet<Arc<str>>, HashSet<Arc<str>>) {
    let mut use_set = HashSet::new();
    let mut def_set = HashSet::new();
    for op in ops {
        match op {
            Opcode::LoadIdent(n) => {
                if !def_set.contains(n) { use_set.insert(n.clone()); }
            }
            Opcode::BindVar(n) | Opcode::StoreVar(n) | Opcode::LetExpr { name: n, .. } => {
                def_set.insert(n.clone());
            }
            _ => {}
        }
    }
    (use_set, def_set)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::v2::vm::Compiler;

    #[test]
    fn cfg_linear_single_block() {
        let p = Compiler::compile_str("1 + 2").unwrap();
        let cfg = Cfg::build(&p);
        assert_eq!(cfg.size(), 1);
    }

    #[test]
    fn cfg_and_creates_branch() {
        let p = Compiler::compile_str("$.a and $.b").unwrap();
        let cfg = Cfg::build(&p);
        assert!(cfg.size() >= 2, "AndOp should create child block");
        let root = &cfg.blocks[cfg.entry];
        assert!(root.branches.iter().any(|e| matches!(e,
            EdgeKind::ShortCircuit { condition: Condition::IfTruthy, .. })));
    }

    #[test]
    fn cfg_filter_creates_loop() {
        let p = Compiler::compile_str("$.x.filter(@.a > 1)").unwrap();
        let cfg = Cfg::build(&p);
        assert!(cfg.size() >= 2);
    }

    #[test]
    fn cfg_reachable_covers_all() {
        let p = Compiler::compile_str("$.a.filter(@.x > 1).map(@.y)").unwrap();
        let cfg = Cfg::build(&p);
        let r = cfg.reachable();
        assert_eq!(r.len(), cfg.size());
    }

    #[test]
    fn cfg_liveness_tracks_let_body() {
        let p = Compiler::compile_str("let x = $.a in x + x").unwrap();
        let cfg = Cfg::build(&p);
        let live = cfg.liveness();
        // Body block should have x live-in.
        let body_has_x = live.live_in.iter().any(|s|
            s.iter().any(|n| n.as_ref() == "x"));
        assert!(body_has_x, "x should be live inside let body");
    }
}
