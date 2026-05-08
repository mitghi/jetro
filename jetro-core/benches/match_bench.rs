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
//! - `demand_map_last`: Large-array late projection into `last()`.
//! - `demand_filter_last`: Large-array selective scan into `last()`.
//! - `demand_sort_take_map`: Sort/take/map chain with bounded ordered demand.
//! - `update_wildcard_batch`: Multi-field functional update over every row.
//! - `update_filtered_batch`: Functional update over a filtered wildcard selector.
//! - `update_unrelated_root_batch`: One root-level batch across unrelated paths.
//! - `update_nested_selected_batch`: Nested selected-object update with shared prefixes.
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

fn make_books_doc() -> Vec<u8> {
    let mut buf = String::from(r#"{"books":["#);
    for i in 0..20_000 {
        if i > 0 {
            buf.push(',');
        }
        let price = (i % 50) + 1;
        let score = 20_000 - i;
        buf.push_str(&format!(
            r#"{{"isbn":"isbn-{i}","price":{price},"score":{score},"name":"book-{i}","tags":["sf"],"tmp":true,"meta":{{"seen":false}}}}"#
        ));
    }
    buf.push_str(r#"],"active":true,"meta":{"updated_at":0,"source":"bench"}}"#);
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

fn demand_map_last(c: &mut Criterion) {
    let doc = make_books_doc();
    c.bench_function("demand_map_last", |b| {
        b.iter(|| {
            let j = Jetro::from_bytes(doc.clone()).expect("parse");
            black_box(j.collect("$.books.map(isbn).last()").expect("eval"))
        })
    });
}

fn demand_filter_last(c: &mut Criterion) {
    let doc = make_books_doc();
    c.bench_function("demand_filter_last", |b| {
        b.iter(|| {
            let j = Jetro::from_bytes(doc.clone()).expect("parse");
            black_box(
                j.collect("$.books.filter(price > 20).map(isbn).last()")
                    .expect("eval"),
            )
        })
    });
}

fn demand_sort_take_map(c: &mut Criterion) {
    let doc = make_books_doc();
    c.bench_function("demand_sort_take_map", |b| {
        b.iter(|| {
            let j = Jetro::from_bytes(doc.clone()).expect("parse");
            black_box(
                j.collect("$.books.sort(-score).take(10).map({isbn, score})")
                    .expect("eval"),
            )
        })
    });
}

fn update_wildcard_batch(c: &mut Criterion) {
    let doc = make_books_doc();
    c.bench_function("update_wildcard_batch", |b| {
        b.iter(|| {
            let j = Jetro::from_bytes(doc.clone()).expect("parse");
            black_box(
                j.collect(
                    r#"$.books[*].update({
                        tags: tags.append("bench"),
                        reviewed: true,
                        tmp: DELETE
                    })"#,
                )
                .expect("eval"),
            )
        })
    });
}

fn update_filtered_batch(c: &mut Criterion) {
    let doc = make_books_doc();
    c.bench_function("update_filtered_batch", |b| {
        b.iter(|| {
            let j = Jetro::from_bytes(doc.clone()).expect("parse");
            black_box(
                j.collect(
                    r#"$.books[* if price > 20].update({
                        tags: tags.append("premium"),
                        "meta.seen": true
                    })"#,
                )
                .expect("eval"),
            )
        })
    });
}

fn update_unrelated_root_batch(c: &mut Criterion) {
    let doc = make_books_doc();
    c.bench_function("update_unrelated_root_batch", |b| {
        b.iter(|| {
            let j = Jetro::from_bytes(doc.clone()).expect("parse");
            black_box(
                j.collect(
                    r#"$.update({
                        active: false,
                        "meta.updated_at": 123,
                        "books[0].tags": @.append("first"),
                        "books[-1].tmp": DELETE
                    })"#,
                )
                .expect("eval"),
            )
        })
    });
}

fn update_nested_selected_batch(c: &mut Criterion) {
    let doc = make_books_doc();
    c.bench_function("update_nested_selected_batch", |b| {
        b.iter(|| {
            let j = Jetro::from_bytes(doc.clone()).expect("parse");
            black_box(
                j.collect(
                    r#"$.books[*].update({
                        "meta.seen": true,
                        "meta.score": score,
                        "meta.label": name
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
    match_range_scan,
    demand_map_last,
    demand_filter_last,
    demand_sort_take_map,
    update_wildcard_batch,
    update_filtered_batch,
    update_unrelated_root_batch,
    update_nested_selected_batch
);
criterion_main!(match_benches);
