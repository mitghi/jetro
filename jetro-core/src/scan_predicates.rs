use crate::ast::{Arg, BinOp, Expr, Step};
use crate::scan::{ScanCmp, ScanPred};

pub(crate) fn canonical_field_eq_literals(args: &[Arg]) -> Option<Vec<(String, Vec<u8>)>> {
    if args.is_empty() || args.len() > 64 {
        return None;
    }
    let mut out = Vec::with_capacity(args.len());
    for a in args {
        let e = match a {
            Arg::Pos(e) | Arg::Named(_, e) => e,
        };
        out.push(canonical_field_eq_literal(e)?);
    }
    Some(out)
}

pub(crate) fn canonical_field_cmp_literal(pred: &Expr) -> Option<(String, ScanCmp, f64)> {
    let (l, op, r) = match pred {
        Expr::BinOp(l, op @ (BinOp::Lt | BinOp::Lte | BinOp::Gt | BinOp::Gte), r) => {
            (&**l, *op, &**r)
        }
        _ => return None,
    };

    let (field, thresh, flip) = match (as_current_field(l), as_num(r)) {
        (Some(f), Some(n)) => (f, n, false),
        _ => match (as_current_field(r), as_num(l)) {
            (Some(f), Some(n)) => (f, n, true),
            _ => return None,
        },
    };

    let scan_op = match (op, flip) {
        (BinOp::Lt, false) | (BinOp::Gt, true) => ScanCmp::Lt,
        (BinOp::Lte, false) | (BinOp::Gte, true) => ScanCmp::Lte,
        (BinOp::Gt, false) | (BinOp::Lt, true) => ScanCmp::Gt,
        (BinOp::Gte, false) | (BinOp::Lte, true) => ScanCmp::Gte,
        _ => return None,
    };
    Some((field, scan_op, thresh))
}

pub(crate) fn canonical_field_mixed_predicates(args: &[Arg]) -> Option<Vec<(String, ScanPred)>> {
    if args.is_empty() || args.len() > 64 {
        return None;
    }
    let mut out = Vec::with_capacity(args.len());
    for a in args {
        let e = match a {
            Arg::Pos(e) | Arg::Named(_, e) => e,
        };
        if let Some((k, lit)) = canonical_field_eq_literal(e) {
            out.push((k, ScanPred::Eq(lit)));
        } else if let Some((k, op, n)) = canonical_field_cmp_literal(e) {
            out.push((k, ScanPred::Cmp(op, n)));
        } else {
            return None;
        }
    }
    Some(out)
}

pub(crate) fn canonical_field_eq_literal(pred: &Expr) -> Option<(String, Vec<u8>)> {
    let (l, r) = match pred {
        Expr::BinOp(l, BinOp::Eq, r) => (&**l, &**r),
        _ => return None,
    };
    let (field, lit) = if let Some(f) = as_current_field(l) {
        (f, r)
    } else if let Some(f) = as_current_field(r) {
        (f, l)
    } else {
        return None;
    };
    let bytes = match lit {
        Expr::Int(n) => n.to_string().into_bytes(),
        Expr::Bool(b) => {
            if *b {
                b"true".to_vec()
            } else {
                b"false".to_vec()
            }
        }
        Expr::Null => b"null".to_vec(),
        Expr::Str(s) => serde_json::to_vec(&serde_json::Value::String(s.clone())).ok()?,
        _ => return None,
    };
    Some((field, bytes))
}

fn as_current_field(e: &Expr) -> Option<String> {
    if let Expr::Chain(base, steps) = e {
        if matches!(**base, Expr::Current) && steps.len() == 1 {
            if let Step::Field(name) = &steps[0] {
                return Some(name.clone());
            }
        }
    }
    None
}

fn as_num(e: &Expr) -> Option<f64> {
    match e {
        Expr::Int(n) => Some(*n as f64),
        Expr::Float(f) => {
            if f.is_finite() {
                Some(*f)
            } else {
                None
            }
        }
        _ => None,
    }
}
