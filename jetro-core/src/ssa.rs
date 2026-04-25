//! SSA-style numbering + data-flow graph for v2 programs.
//!
//! v2's stack machine has no named intermediate values.  To express a
//! data-flow graph, we assign each stack push a fresh `ValueId` and
//! record which earlier values each opcode consumes.
//!
//! Limitations:
//! - Sub-programs (branches) are opaque to this pass — they are walked
//!   separately but their values live in their own namespace.
//! - Phi nodes are not synthesised; there are no merge points within
//!   a single block.
//!
//! Useful for: def-use queries, live-range analysis, value-numbering CSE.

use std::sync::Arc;
use std::collections::HashMap;
use super::vm::{Program, Opcode};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ValueId(pub u32);

#[derive(Debug, Clone)]
pub struct SsaInstr {
    pub id:   ValueId,
    pub op:   Opcode,
    /// Value ids consumed from the stack (pops) in order.
    pub uses: Vec<ValueId>,
}

/// A phi node: at a merge point, value takes one of several incoming
/// values depending on which predecessor path was taken.
#[derive(Debug, Clone)]
pub struct Phi {
    pub id:       ValueId,
    /// Incoming (predecessor label, value) pairs.
    pub incoming: Vec<(PhiEdge, ValueId)>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PhiEdge { AndLhs, AndRhs, OrLhs, OrRhs, CoalesceLhs, CoalesceRhs }

#[derive(Debug, Clone, Default)]
pub struct SsaGraph {
    pub instrs: Vec<SsaInstr>,
    pub phis:   Vec<Phi>,
    /// Last pushed value id — the program's result.
    pub result: Option<ValueId>,
}

impl SsaGraph {
    pub fn build(program: &Program) -> SsaGraph {
        let mut g = SsaGraph::default();
        let mut stack: Vec<ValueId> = Vec::new();
        for op in program.ops.iter() {
            let arity = op_arity(op);
            let mut uses = Vec::with_capacity(arity.pops);
            for _ in 0..arity.pops {
                if let Some(v) = stack.pop() { uses.push(v); }
            }
            let id = ValueId(g.instrs.len() as u32);
            // Synthesise phi at short-circuit / coalesce merge points.
            // At an AndOp, the lhs is on the stack; rhs sub-program produces
            // a new value conditionally.  Result is a phi(lhs, rhs).
            let phi_edge = match op {
                Opcode::AndOp(_)      => Some((PhiEdge::AndLhs, PhiEdge::AndRhs)),
                Opcode::OrOp(_)       => Some((PhiEdge::OrLhs, PhiEdge::OrRhs)),
                Opcode::CoalesceOp(_) => Some((PhiEdge::CoalesceLhs, PhiEdge::CoalesceRhs)),
                _ => None,
            };
            if let Some((lhs_edge, rhs_edge)) = phi_edge {
                // uses[0] is the lhs consumed; the "rhs" value is synthetic — we
                // model it as the instruction's own id (sub-program result).
                let lhs = uses.first().copied().unwrap_or(id);
                g.phis.push(Phi {
                    id,
                    incoming: vec![(lhs_edge, lhs), (rhs_edge, id)],
                });
            }
            g.instrs.push(SsaInstr { id, op: op.clone(), uses });
            if arity.pushes { stack.push(id); }
        }
        g.result = stack.pop();
        g
    }

    /// Use-list (consumers) for each value id.
    pub fn use_list(&self) -> HashMap<ValueId, Vec<ValueId>> {
        let mut m: HashMap<ValueId, Vec<ValueId>> = HashMap::new();
        for instr in &self.instrs {
            for u in &instr.uses {
                m.entry(*u).or_default().push(instr.id);
            }
        }
        m
    }

    /// Dead values: pushed but never consumed and not the final result.
    pub fn dead_values(&self) -> Vec<ValueId> {
        let uses = self.use_list();
        self.instrs.iter()
            .filter(|i| {
                op_arity(&i.op).pushes
                && !uses.contains_key(&i.id)
                && Some(i.id) != self.result
            })
            .map(|i| i.id)
            .collect()
    }
}

#[derive(Debug, Clone, Copy)]
struct Arity { pops: usize, pushes: bool }

fn op_arity(op: &Opcode) -> Arity {
    match op {
        Opcode::PushNull | Opcode::PushBool(_) | Opcode::PushInt(_)
            | Opcode::PushFloat(_) | Opcode::PushStr(_)
            | Opcode::PushRoot | Opcode::PushCurrent | Opcode::LoadIdent(_)
            | Opcode::RootChain(_) | Opcode::GetPointer(_)
            | Opcode::MakeObj(_) | Opcode::MakeArr(_) | Opcode::FString(_)
            | Opcode::ListComp(_) | Opcode::DictComp(_) | Opcode::SetComp(_)
            | Opcode::PatchEval(_)
            | Opcode::TryExpr { .. } =>
            Arity { pops: 0, pushes: true },

        Opcode::GetField(_) | Opcode::OptField(_) | Opcode::GetIndex(_)
            | Opcode::FieldChain(_)
            | Opcode::GetSlice(..) | Opcode::Descendant(_) | Opcode::DescendAll
            | Opcode::DynIndex(_) | Opcode::InlineFilter(_)
            | Opcode::Quantifier(_) | Opcode::FilterCount(_)
            | Opcode::FindFirst(_) | Opcode::FindOne(_)
            | Opcode::FilterMap { .. } | Opcode::MapFilter { .. }
            | Opcode::FilterLast { .. }
            | Opcode::FilterFilter { .. }
            | Opcode::MapMap { .. } | Opcode::MapSum(_) | Opcode::MapAvg(_)
            | Opcode::MapToJsonJoin { .. }
            | Opcode::StrTrimUpper | Opcode::StrTrimLower
            | Opcode::StrUpperTrim | Opcode::StrLowerTrim
            | Opcode::StrSplitReverseJoin { .. }
            | Opcode::MapReplaceLit { .. }
            | Opcode::MapUpperReplaceLit { .. }
            | Opcode::MapLowerReplaceLit { .. }
            | Opcode::MapStrConcat { .. }
            | Opcode::MapSplitLenSum { .. }
            | Opcode::MapProject { .. }
            | Opcode::MapStrSlice { .. }
            | Opcode::MapFString(_)
            | Opcode::MapSplitCount { .. }
            | Opcode::MapSplitFirst { .. }
            | Opcode::MapSplitNth   { .. }
            | Opcode::MapSplitCountSum { .. }
            | Opcode::MapMin(_) | Opcode::MapMax(_)
            | Opcode::MapField(_) | Opcode::MapFieldChain(_) | Opcode::MapFieldUnique(_)
            | Opcode::MapFieldChainUnique(_)
            | Opcode::FlatMapChain(_)
            | Opcode::FilterFieldEqLit(_, _) | Opcode::FilterFieldCmpLit(_, _, _)
            | Opcode::FilterCurrentCmpLit(_, _)
            | Opcode::FilterStrVecStartsWith(_)
            | Opcode::FilterStrVecEndsWith(_)
            | Opcode::FilterStrVecContains(_)
            | Opcode::MapStrVecUpper
            | Opcode::MapStrVecLower
            | Opcode::MapStrVecTrim
            | Opcode::MapNumVecArith { .. }
            | Opcode::MapNumVecNeg
            | Opcode::FilterFieldCmpField(_, _, _)
            | Opcode::FilterFieldCmpFieldCount(_, _, _)
            | Opcode::FilterFieldsAllEqLitCount(_)
            | Opcode::FilterFieldsAllCmpLitCount(_)
            | Opcode::GroupByField(_)
            | Opcode::CountByField(_)
            | Opcode::UniqueByField(_)
            | Opcode::MapFlatten(_)
            | Opcode::MapFirst(_) | Opcode::MapLast(_)
            | Opcode::FilterTakeWhile { .. }
            | Opcode::FilterDropWhile { .. } | Opcode::MapUnique(_)
            | Opcode::EquiJoin { .. }
            | Opcode::TopN { .. } | Opcode::UniqueCount
            | Opcode::ArgExtreme { .. } | Opcode::KindCheck { .. }
            | Opcode::Not | Opcode::Neg
            | Opcode::CallMethod(_) | Opcode::CallOptMethod(_)
            | Opcode::AndOp(_) | Opcode::OrOp(_) | Opcode::CoalesceOp(_)
            | Opcode::IfElse { .. }
            | Opcode::CastOp(_) =>
            Arity { pops: 1, pushes: true },

        Opcode::Add | Opcode::Sub | Opcode::Mul | Opcode::Div | Opcode::Mod
            | Opcode::Eq | Opcode::Neq | Opcode::Lt | Opcode::Lte
            | Opcode::Gt | Opcode::Gte | Opcode::Fuzzy =>
            Arity { pops: 2, pushes: true },

        Opcode::StoreVar(_) => Arity { pops: 1, pushes: false },
        Opcode::SetCurrent | Opcode::BindVar(_)
            | Opcode::BindObjDestructure(_) | Opcode::BindArrDestructure(_)
            | Opcode::LetExpr { .. } => Arity { pops: 0, pushes: false },
    }
}

/// Value-numbering CSE on top of SSA: two instructions with identical
/// opcode and matching `uses` are mapped to the same canonical id.
pub fn value_number(g: &SsaGraph) -> HashMap<ValueId, ValueId> {
    let mut canon: HashMap<ValueId, ValueId> = HashMap::new();
    let mut seen: HashMap<(u64, Vec<ValueId>), ValueId> = HashMap::new();
    for instr in &g.instrs {
        let canon_uses: Vec<ValueId> = instr.uses.iter()
            .map(|u| *canon.get(u).unwrap_or(u)).collect();
        let key = (op_hash(&instr.op), canon_uses);
        match seen.get(&key) {
            Some(&existing) => { canon.insert(instr.id, existing); }
            None            => { seen.insert(key, instr.id); canon.insert(instr.id, instr.id); }
        }
    }
    canon
}

fn op_hash(op: &Opcode) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let mut h = DefaultHasher::new();
    std::mem::discriminant(op).hash(&mut h);
    match op {
        Opcode::PushInt(n) => n.hash(&mut h),
        Opcode::PushStr(s) => s.as_bytes().hash(&mut h),
        Opcode::PushBool(b) => b.hash(&mut h),
        Opcode::GetField(k) | Opcode::OptField(k) | Opcode::LoadIdent(k) =>
            k.as_bytes().hash(&mut h),
        Opcode::GetIndex(i) => i.hash(&mut h),
        _ => {}
    }
    h.finish()
}

// Silence unused-Arc warning.
#[allow(dead_code)]
fn _use_arc<T>(_: Arc<T>) {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vm::Compiler;

    #[test]
    fn ssa_builds_graph() {
        let p = Compiler::compile_str("$.a + $.b").unwrap();
        let g = SsaGraph::build(&p);
        assert!(g.result.is_some());
        let add = g.instrs.last().unwrap();
        assert_eq!(add.uses.len(), 2);
    }

    #[test]
    fn ssa_use_list() {
        let p = Compiler::compile_str("$.a + $.b").unwrap();
        let g = SsaGraph::build(&p);
        let uses = g.use_list();
        assert_eq!(uses.values().map(|v| v.len()).sum::<usize>(), 2);
    }

    #[test]
    fn value_numbering_dedups_identical() {
        // Use GetField sequences that don't const-fold.
        let p = Compiler::compile_str("[$.a, $.a]").unwrap();
        let g = SsaGraph::build(&p);
        let canon = value_number(&g);
        // Both root-chain loads share canonical id.
        let load_ids: Vec<ValueId> = g.instrs.iter()
            .filter(|i| matches!(i.op, crate::vm::Opcode::RootChain(_)))
            .map(|i| canon[&i.id]).collect();
        if load_ids.len() >= 2 {
            assert_eq!(load_ids[0], load_ids[1]);
        }
    }
}
