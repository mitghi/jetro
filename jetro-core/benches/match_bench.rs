//! Microbenchmarks for the `match` runtime.
//!
//! Workloads:
//! - `match_filter_take`: Take the first three rows that satisfy a
//!   single-arm match predicate, exercising the demand cutoff path.
//! - `match_dispatch_5_arms`: Tagged-union dispatch across five arms
//!   over a thousand-row array, exercising the cross-arm shared-prefix
//!   optimisation.
//! - `match_range_scan`: Range-pattern dispatch over an integer array,
//!   exercising the `RangeCheck` opcode and `val_to_f64` widening.
//!
//! Run with `cargo bench -p jetro-core --bench match_bench`. Each
//! benchmark builds a fresh `Jetro` document per iteration so the
//! reported numbers are end-to-end (parse + compile + execute), not
//! steady-state. Add a baseline with
//! `cargo bench -p jetro-core --bench match_bench -- --save-baseline current`
//! to compare future runs.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use jetro_core::Jetro;

fn make_filter_doc() -> Vec<u8> {
    let mut buf = String::from(r#"{"xs":["#);
    for i in 0..1024 {
        if i > 0 {
            buf.push(',');
        }
        buf.push_str(&i.to_string());
    }
    buf.push_str("]}");
    buf.into_bytes()
}

fn make_dispatch_doc() -> Vec<u8> {
    let mut buf = String::from(r#"{"events":["#);
    let tags = ["view", "click", "submit", "error", "drop"];
    for i in 0..1024 {
        if i > 0 {
            buf.push(',');
        }
        let tag = tags[i % tags.len()];
        buf.push_str(&format!(r#"{{"tag":"{tag}","id":{i}}}"#));
    }
    buf.push_str("]}");
    buf.into_bytes()
}

fn match_filter_take(c: &mut Criterion) {
    let doc = make_filter_doc();
    c.bench_function("match_filter_take", |b| {
        b.iter(|| {
            let j = Jetro::from_bytes(doc.clone()).expect("parse");
            black_box(
                j.collect(
                    r#"$.xs.filter(match @ with {
                        n when n > 100 -> true,
                        _              -> false
                    }).take(3)"#,
                )
                .expect("eval"),
            )
        })
    });
}

fn match_dispatch_5_arms(c: &mut Criterion) {
    let doc = make_dispatch_doc();
    c.bench_function("match_dispatch_5_arms", |b| {
        b.iter(|| {
            let j = Jetro::from_bytes(doc.clone()).expect("parse");
            black_box(
                j.collect(
                    r#"$.events.map(match @ with {
                        {tag: "view",   id: i} -> {sort: 0, n: i},
                        {tag: "click",  id: i} -> {sort: 1, n: i},
                        {tag: "submit", id: i} -> {sort: 2, n: i},
                        {tag: "error",  id: i} -> {sort: 3, n: i},
                        _                      -> {sort: 4, n: 0}
                    })"#,
                )
                .expect("eval"),
            )
        })
    });
}

fn match_range_scan(c: &mut Criterion) {
    let doc = make_filter_doc();
    c.bench_function("match_range_scan", |b| {
        b.iter(|| {
            let j = Jetro::from_bytes(doc.clone()).expect("parse");
            black_box(
                j.collect(
                    r#"$.xs.map(match @ with {
                        0..=100      -> "small",
                        101..=500    -> "medium",
                        _            -> "large"
                    })"#,
                )
                .expect("eval"),
            )
        })
    });
}

criterion_group!(
    match_benches,
    match_filter_take,
    match_dispatch_5_arms,
    match_range_scan
);
criterion_main!(match_benches);
