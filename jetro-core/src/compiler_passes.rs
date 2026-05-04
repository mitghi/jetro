//! Peephole optimization passes for the bytecode compiler.
//!
//! Each pass takes a `Vec<Opcode>` and returns a transformed `Vec<Opcode>`.
//! Passes are called from `Compiler::optimize_with` in `compiler.rs`.

use std::sync::Arc;

use crate::ast::Arg;
use crate::builtins::BuiltinMethod;
use crate::vm::{
    CompiledCall, FieldChainData, Opcode, Program,
    disable_opcode_fusion,
};

fn make_noarg_call(method: BuiltinMethod, name: &str) -> Opcode {
    Opcode::CallMethod(Arc::new(CompiledCall {
        method,
        name: Arc::from(name),
        sub_progs: Arc::from(&[] as &[Arc<Program>]),
        orig_args: Arc::from(&[] as &[Arg]),
        demand_max_keep: None,
    }))
}

/// Demand-annotation pass: when a `filter` or `map` is immediately followed by
/// `take(n)`, annotate the call's `demand_max_keep` so the inner loop stops early.
pub(crate) fn pass_method_demand(ops: Vec<Opcode>) -> Vec<Opcode> {
    fn take_const(call: &CompiledCall) -> Option<usize> {
        use crate::ast::Expr;
        if call.name.as_ref() != "take" {
            return None;
        }
        if call.orig_args.len() != 1 {
            return None;
        }
        match &call.orig_args[0] {
            Arg::Pos(Expr::Int(n)) if *n >= 0 => Some(*n as usize),
            _ => None,
        }
    }

    fn is_demand_aware(method: BuiltinMethod) -> bool {
        matches!(method, BuiltinMethod::Filter | BuiltinMethod::Map)
    }
    let mut out = Vec::with_capacity(ops.len());
    let mut i = 0;
    while i < ops.len() {
        if i + 1 < ops.len() {
            if let (Opcode::CallMethod(a), Opcode::CallMethod(b)) = (&ops[i], &ops[i + 1]) {
                if is_demand_aware(a.method) && a.demand_max_keep.is_none() {
                    if let Some(n) = take_const(b) {
                        let mut new_call = (**a).clone();
                        new_call.demand_max_keep = Some(n);
                        out.push(Opcode::CallMethod(Arc::new(new_call)));
                        i += 2;
                        continue;
                    }
                }
            }
        }
        out.push(ops[i].clone());
        i += 1;
    }
    out
}

/// Replace `OptField` with `GetField` when the preceding opcode always produces
/// a non-null value (e.g. `MakeObj`), eliminating the null-propagation overhead.
pub(crate) fn pass_nullness_opt_field(ops: Vec<Opcode>) -> Vec<Opcode> {
    let mut out: Vec<Opcode> = Vec::with_capacity(ops.len());
    for op in ops {
        if let Opcode::OptField(k) = &op {
            let non_null = matches!(out.last(), Some(Opcode::MakeObj(_)));
            if non_null {
                out.push(Opcode::GetField(k.clone()));
                continue;
            }
        }
        out.push(op);
    }
    out
}

/// Fold no-argument method calls on constant operands (e.g. `"hello".len()` → `5`).
/// Covers `len`, `upper`, `lower`, `trim` on string literals and `len` on non-spread arrays.
pub(crate) fn pass_method_const_fold(ops: Vec<Opcode>) -> Vec<Opcode> {
    let mut out: Vec<Opcode> = Vec::with_capacity(ops.len());
    for op in ops {
        if let Opcode::CallMethod(c) = &op {
            if c.sub_progs.is_empty() {
                match (out.last(), c.method) {
                    (Some(Opcode::PushStr(s)), BuiltinMethod::Len) => {
                        let n = s.chars().count() as i64;
                        out.pop();
                        out.push(Opcode::PushInt(n));
                        continue;
                    }
                    (Some(Opcode::PushStr(s)), BuiltinMethod::Upper) => {
                        let u: Arc<str> = Arc::from(s.to_uppercase());
                        out.pop();
                        out.push(Opcode::PushStr(u));
                        continue;
                    }
                    (Some(Opcode::PushStr(s)), BuiltinMethod::Lower) => {
                        let u: Arc<str> = Arc::from(s.to_lowercase());
                        out.pop();
                        out.push(Opcode::PushStr(u));
                        continue;
                    }
                    (Some(Opcode::PushStr(s)), BuiltinMethod::Trim) => {
                        let u: Arc<str> = Arc::from(s.trim());
                        out.pop();
                        out.push(Opcode::PushStr(u));
                        continue;
                    }
                    (Some(Opcode::MakeArr(progs)), BuiltinMethod::Len) => {
                        if progs.iter().all(|(_, sp)| !*sp) {
                            let n = progs.len() as i64;
                            out.pop();
                            out.push(Opcode::PushInt(n));
                            continue;
                        }
                    }
                    _ => {}
                }
            }
        }
        out.push(op);
    }
    out
}

/// Fold `KindCheck` against a preceding literal push into a constant boolean,
/// e.g. `MakeObj` followed by `is array` → `false`.
pub(crate) fn pass_kind_check_fold(ops: Vec<Opcode>) -> Vec<Opcode> {
    use crate::analysis::{fold_kind_check, VType};
    let mut out = Vec::with_capacity(ops.len());
    for op in ops {
        if let Opcode::KindCheck { ty, negate } = &op {
            let prev_ty: Option<VType> = match out.last() {
                Some(Opcode::PushNull) => Some(VType::Null),
                Some(Opcode::PushBool(_)) => Some(VType::Bool),
                Some(Opcode::PushInt(_)) => Some(VType::Int),
                Some(Opcode::PushFloat(_)) => Some(VType::Float),
                Some(Opcode::PushStr(_)) => Some(VType::Str),
                Some(Opcode::MakeArr(_)) => Some(VType::Arr),
                Some(Opcode::MakeObj(_)) => Some(VType::Obj),
                _ => None,
            };
            if let Some(vt) = prev_ty {
                if let Some(b) = fold_kind_check(vt, *ty, *negate) {
                    out.pop();
                    out.push(Opcode::PushBool(b));
                    continue;
                }
            }
        }
        out.push(op);
    }
    out
}

/// Placeholder for filter/field specialisation fusion; currently a pass-through
/// that preserves opcodes for future pattern-specific fast paths.
pub(crate) fn pass_field_specialise(ops: Vec<Opcode>) -> Vec<Opcode> {
    if disable_opcode_fusion() {
        return ops;
    }
    let mut out2: Vec<Opcode> = Vec::with_capacity(ops.len());
    for op in ops {
        match op {
            Opcode::CallMethod(ref b) => {
                let _ = b;
            }
            _ => {}
        }
        out2.push(op);
    }
    let mut out3: Vec<Opcode> = Vec::with_capacity(out2.len());
    for op in out2 {
        out3.push(op);
    }
    out3
}

/// Placeholder for list-comprehension specialisation; currently a pass-through.
pub(crate) fn pass_list_comp_specialise(ops: Vec<Opcode>) -> Vec<Opcode> {
    ops
}

/// Strength-reduction pass: replace expensive method sequences with cheaper equivalents.
/// Examples: `sort()[0]` → `min()`, `reverse().first()` → `last()`, `sort().sort()` → `sort()`.
pub(crate) fn pass_strength_reduce(ops: Vec<Opcode>) -> Vec<Opcode> {
    let mut out: Vec<Opcode> = Vec::with_capacity(ops.len());
    for op in ops {
        if let Some(Opcode::CallMethod(prev)) = out.last().cloned() {
            let replaced = match (prev.method, &op) {
                (BuiltinMethod::Sort, Opcode::GetIndex(0)) if prev.sub_progs.is_empty() => {
                    Some(make_noarg_call(BuiltinMethod::Min, "min"))
                }
                (BuiltinMethod::Sort, Opcode::GetIndex(-1)) if prev.sub_progs.is_empty() => {
                    Some(make_noarg_call(BuiltinMethod::Max, "max"))
                }
                (BuiltinMethod::Sort, Opcode::CallMethod(next))
                    if prev.sub_progs.is_empty() && next.method == BuiltinMethod::First =>
                {
                    Some(make_noarg_call(BuiltinMethod::Min, "min"))
                }
                (BuiltinMethod::Sort, Opcode::CallMethod(next))
                    if prev.sub_progs.is_empty() && next.method == BuiltinMethod::Last =>
                {
                    Some(make_noarg_call(BuiltinMethod::Max, "max"))
                }
                (BuiltinMethod::Reverse, Opcode::CallMethod(next))
                    if next.method == BuiltinMethod::First =>
                {
                    Some(make_noarg_call(BuiltinMethod::Last, "last"))
                }
                (BuiltinMethod::Reverse, Opcode::CallMethod(next))
                    if next.method == BuiltinMethod::Last =>
                {
                    Some(make_noarg_call(BuiltinMethod::First, "first"))
                }
                (
                    BuiltinMethod::Sort | BuiltinMethod::Reverse | BuiltinMethod::Map,
                    Opcode::CallMethod(next),
                ) if next.sub_progs.is_empty()
                    && (next.method == BuiltinMethod::Len
                        || next.method == BuiltinMethod::Count) =>
                {
                    Some(Opcode::CallMethod(Arc::clone(next)))
                }
                (BuiltinMethod::Sort | BuiltinMethod::Reverse, Opcode::CallMethod(next))
                    if prev.sub_progs.is_empty()
                        && next.sub_progs.is_empty()
                        && matches!(
                            next.method,
                            BuiltinMethod::Sum
                                | BuiltinMethod::Avg
                                | BuiltinMethod::Min
                                | BuiltinMethod::Max
                        ) =>
                {
                    Some(Opcode::CallMethod(Arc::clone(next)))
                }
                (BuiltinMethod::Sort, Opcode::CallMethod(next))
                    if prev.sub_progs.is_empty()
                        && next.method == BuiltinMethod::Sort
                        && next.sub_progs.is_empty() =>
                {
                    Some(Opcode::CallMethod(Arc::clone(next)))
                }
                (BuiltinMethod::Unique, Opcode::CallMethod(next))
                    if next.method == BuiltinMethod::Unique =>
                {
                    Some(Opcode::CallMethod(Arc::clone(next)))
                }
                _ => None,
            };
            if let Some(rep) = replaced {
                out.pop();
                out.push(rep);
                continue;
            }
            if prev.method == BuiltinMethod::Reverse && prev.sub_progs.is_empty() {
                if let Opcode::CallMethod(next) = &op {
                    if next.method == BuiltinMethod::Reverse && next.sub_progs.is_empty() {
                        out.pop();
                        continue;
                    }
                }
            }
        }
        out.push(op);
    }
    out
}

/// Fuse runs of two or more consecutive `GetField`/`OptField` opcodes into a
/// single `FieldChain` opcode, reducing dispatch overhead and enabling per-step ICs.
pub(crate) fn pass_field_chain(ops: Vec<Opcode>) -> Vec<Opcode> {
    fn field_key(op: &Opcode) -> Option<Arc<str>> {
        match op {
            Opcode::GetField(k) | Opcode::OptField(k) => Some(Arc::clone(k)),
            _ => None,
        }
    }
    let mut out = Vec::with_capacity(ops.len());
    let mut it = ops.into_iter().peekable();
    while let Some(op) = it.next() {
        if let Some(k0) = field_key(&op) {
            if it.peek().and_then(field_key).is_some() {
                let mut chain: Vec<Arc<str>> = vec![k0];
                while let Some(k) = it.peek().and_then(field_key) {
                    it.next();
                    chain.push(k);
                }
                out.push(Opcode::FieldChain(Arc::new(FieldChainData::new(chain.into()))));
                continue;
            }
            out.push(op);
        } else {
            out.push(op);
        }
    }
    out
}

/// Fuse `PushRoot` followed by one or more `GetField` opcodes into a single
/// `RootChain` opcode, enabling path-cache lookups without individual stack pushes.
pub(crate) fn pass_root_chain(ops: Vec<Opcode>) -> Vec<Opcode> {
    let mut out = Vec::with_capacity(ops.len());
    let mut it = ops.into_iter().peekable();
    while let Some(op) = it.next() {
        if matches!(op, Opcode::PushRoot) {
            let mut chain: Vec<Arc<str>> = Vec::new();
            while let Some(Opcode::GetField(_)) = it.peek() {
                if let Some(Opcode::GetField(k)) = it.next() {
                    chain.push(k);
                }
            }
            if chain.is_empty() {
                out.push(Opcode::PushRoot);
            } else {
                out.push(Opcode::RootChain(chain.into()));
            }
        } else {
            out.push(op);
        }
    }
    out
}

/// Eliminate provably redundant adjacent opcodes: `reverse().reverse()` → identity,
/// `!!` → identity, double `unique`/`compact`/`sort`, consecutive quantifiers, etc.
pub(crate) fn pass_redundant_ops(ops: Vec<Opcode>) -> Vec<Opcode> {
    let mut out: Vec<Opcode> = Vec::with_capacity(ops.len());
    for op in ops {
        match (&op, out.last()) {
            (Opcode::CallMethod(b), Some(Opcode::CallMethod(a)))
                if a.method == BuiltinMethod::Reverse && b.method == BuiltinMethod::Reverse =>
            {
                out.pop();
                continue;
            }
            (Opcode::CallMethod(b), Some(Opcode::CallMethod(a)))
                if a.method == b.method
                    && matches!(a.method, BuiltinMethod::Unique | BuiltinMethod::Compact)
                    && a.sub_progs.is_empty()
                    && b.sub_progs.is_empty() =>
            {
                out.pop();
                out.push(op);
                continue;
            }
            (Opcode::CallMethod(b), Some(Opcode::CallMethod(a)))
                if a.method == BuiltinMethod::Sort && b.method == BuiltinMethod::Sort =>
            {
                out.pop();
                out.push(op);
                continue;
            }
            (Opcode::Quantifier(_), Some(Opcode::Quantifier(_))) => {
                out.pop();
                out.push(op);
                continue;
            }
            (Opcode::Not, Some(Opcode::Not)) => {
                out.pop();
                continue;
            }
            (Opcode::Neg, Some(Opcode::Neg)) => {
                out.pop();
                continue;
            }
            _ => {}
        }
        out.push(op);
    }
    out
}

/// Constant-fold arithmetic, comparison, and logical opcodes when all operands
/// are known literals, reducing runtime work to a single `Push` opcode.
pub(crate) fn pass_const_fold(ops: Vec<Opcode>) -> Vec<Opcode> {
    let mut out = Vec::with_capacity(ops.len());
    let mut i = 0;
    while i < ops.len() {
        if i + 1 < ops.len() {
            let folded = match (&ops[i], &ops[i + 1]) {
                (Opcode::PushBool(false), Opcode::AndOp(_)) => Some(Opcode::PushBool(false)),
                (Opcode::PushBool(true), Opcode::OrOp(_)) => Some(Opcode::PushBool(true)),
                _ => None,
            };
            if let Some(folded) = folded {
                out.push(folded);
                i += 2;
                continue;
            }
        }
        if i + 1 < ops.len() {
            let folded = match (&ops[i], &ops[i + 1]) {
                (Opcode::PushBool(b), Opcode::Not) => Some(Opcode::PushBool(!b)),
                (Opcode::PushInt(n), Opcode::Neg) => Some(Opcode::PushInt(-n)),
                (Opcode::PushFloat(f), Opcode::Neg) => Some(Opcode::PushFloat(-f)),
                _ => None,
            };
            if let Some(folded) = folded {
                out.push(folded);
                i += 2;
                continue;
            }
        }
        if i + 2 < ops.len() {
            let folded = match (&ops[i], &ops[i + 1], &ops[i + 2]) {
                (Opcode::PushInt(a), Opcode::PushInt(b), Opcode::Add) => Some(Opcode::PushInt(a + b)),
                (Opcode::PushInt(a), Opcode::PushInt(b), Opcode::Sub) => Some(Opcode::PushInt(a - b)),
                (Opcode::PushInt(a), Opcode::PushInt(b), Opcode::Mul) => Some(Opcode::PushInt(a * b)),
                (Opcode::PushInt(a), Opcode::PushInt(b), Opcode::Mod) if *b != 0 => Some(Opcode::PushInt(a % b)),
                (Opcode::PushInt(a), Opcode::PushInt(b), Opcode::Div) if *b != 0 => Some(Opcode::PushFloat(*a as f64 / *b as f64)),
                (Opcode::PushFloat(a), Opcode::PushFloat(b), Opcode::Add) => Some(Opcode::PushFloat(a + b)),
                (Opcode::PushFloat(a), Opcode::PushFloat(b), Opcode::Sub) => Some(Opcode::PushFloat(a - b)),
                (Opcode::PushFloat(a), Opcode::PushFloat(b), Opcode::Mul) => Some(Opcode::PushFloat(a * b)),
                (Opcode::PushFloat(a), Opcode::PushFloat(b), Opcode::Div) if *b != 0.0 => Some(Opcode::PushFloat(a / b)),
                (Opcode::PushInt(a), Opcode::PushFloat(b), Opcode::Add) => Some(Opcode::PushFloat(*a as f64 + b)),
                (Opcode::PushFloat(a), Opcode::PushInt(b), Opcode::Add) => Some(Opcode::PushFloat(a + *b as f64)),
                (Opcode::PushInt(a), Opcode::PushFloat(b), Opcode::Sub) => Some(Opcode::PushFloat(*a as f64 - b)),
                (Opcode::PushFloat(a), Opcode::PushInt(b), Opcode::Sub) => Some(Opcode::PushFloat(a - *b as f64)),
                (Opcode::PushInt(a), Opcode::PushFloat(b), Opcode::Mul) => Some(Opcode::PushFloat(*a as f64 * b)),
                (Opcode::PushFloat(a), Opcode::PushInt(b), Opcode::Mul) => Some(Opcode::PushFloat(a * *b as f64)),
                (Opcode::PushInt(a), Opcode::PushFloat(b), Opcode::Div) if *b != 0.0 => Some(Opcode::PushFloat(*a as f64 / b)),
                (Opcode::PushFloat(a), Opcode::PushInt(b), Opcode::Div) if *b != 0 => Some(Opcode::PushFloat(a / *b as f64)),
                (Opcode::PushInt(a), Opcode::PushFloat(b), Opcode::Lt) => Some(Opcode::PushBool((*a as f64) < *b)),
                (Opcode::PushFloat(a), Opcode::PushInt(b), Opcode::Lt) => Some(Opcode::PushBool(*a < (*b as f64))),
                (Opcode::PushInt(a), Opcode::PushFloat(b), Opcode::Gt) => Some(Opcode::PushBool((*a as f64) > *b)),
                (Opcode::PushFloat(a), Opcode::PushInt(b), Opcode::Gt) => Some(Opcode::PushBool(*a > (*b as f64))),
                (Opcode::PushInt(a), Opcode::PushFloat(b), Opcode::Lte) => Some(Opcode::PushBool((*a as f64) <= *b)),
                (Opcode::PushFloat(a), Opcode::PushInt(b), Opcode::Lte) => Some(Opcode::PushBool(*a <= (*b as f64))),
                (Opcode::PushInt(a), Opcode::PushFloat(b), Opcode::Gte) => Some(Opcode::PushBool((*a as f64) >= *b)),
                (Opcode::PushFloat(a), Opcode::PushInt(b), Opcode::Gte) => Some(Opcode::PushBool(*a >= (*b as f64))),
                (Opcode::PushFloat(a), Opcode::PushFloat(b), Opcode::Lt) => Some(Opcode::PushBool(a < b)),
                (Opcode::PushFloat(a), Opcode::PushFloat(b), Opcode::Lte) => Some(Opcode::PushBool(a <= b)),
                (Opcode::PushFloat(a), Opcode::PushFloat(b), Opcode::Gt) => Some(Opcode::PushBool(a > b)),
                (Opcode::PushFloat(a), Opcode::PushFloat(b), Opcode::Gte) => Some(Opcode::PushBool(a >= b)),
                (Opcode::PushInt(a), Opcode::PushInt(b), Opcode::Eq) => Some(Opcode::PushBool(a == b)),
                (Opcode::PushInt(a), Opcode::PushInt(b), Opcode::Neq) => Some(Opcode::PushBool(a != b)),
                (Opcode::PushInt(a), Opcode::PushInt(b), Opcode::Lt) => Some(Opcode::PushBool(a < b)),
                (Opcode::PushInt(a), Opcode::PushInt(b), Opcode::Lte) => Some(Opcode::PushBool(a <= b)),
                (Opcode::PushInt(a), Opcode::PushInt(b), Opcode::Gt) => Some(Opcode::PushBool(a > b)),
                (Opcode::PushInt(a), Opcode::PushInt(b), Opcode::Gte) => Some(Opcode::PushBool(a >= b)),
                (Opcode::PushStr(a), Opcode::PushStr(b), Opcode::Eq) => Some(Opcode::PushBool(a == b)),
                (Opcode::PushStr(a), Opcode::PushStr(b), Opcode::Neq) => Some(Opcode::PushBool(a != b)),
                (Opcode::PushStr(a), Opcode::PushStr(b), Opcode::Lt) => Some(Opcode::PushBool(a < b)),
                (Opcode::PushStr(a), Opcode::PushStr(b), Opcode::Lte) => Some(Opcode::PushBool(a <= b)),
                (Opcode::PushStr(a), Opcode::PushStr(b), Opcode::Gt) => Some(Opcode::PushBool(a > b)),
                (Opcode::PushStr(a), Opcode::PushStr(b), Opcode::Gte) => Some(Opcode::PushBool(a >= b)),
                (Opcode::PushStr(a), Opcode::PushStr(b), Opcode::Add) => {
                    Some(Opcode::PushStr(Arc::<str>::from(format!("{}{}", a, b))))
                }
                (Opcode::PushBool(a), Opcode::PushBool(b), Opcode::Eq) => Some(Opcode::PushBool(a == b)),
                _ => None,
            };
            if let Some(folded) = folded {
                out.push(folded);
                i += 3;
                continue;
            }
        }
        out.push(ops[i].clone());
        i += 1;
    }
    out
}
