//! Head-to-head: jetro vs jaq (https://github.com/01mf02/jaq) in-process,
//! against the same semi-large synthetic payload used by bench_complex.
//!
//! Run:
//!   cargo run --release --example bench_vs_jaq -p jetro-core
//!
//! Both engines parse the JSON once and execute N warmed iterations of
//! each query.  We report jetro tree-walker, jetro byte-scan (where it
//! applies), and jaq — best/median/mean per engine + relative speedup.
//!
//! Notes:
//! - Jetro and jq syntaxes differ; we hand-translate each query to the
//!   most idiomatic equivalent.  If the jq form is structurally weaker
//!   (e.g. wraps with `[.. | objects | select(...)]`) we keep it — the
//!   goal is "typical user formulation", not identical IR.
//! - jaq's `Val` is built from serde_json once (outside the timed
//!   section), just like jetro caches its `Val` via `OnceCell`.
//! - For jetro we execute via `Jetro::collect` which routes through the
//!   thread-local VM with compile-cache + IC hot.

use jetro_core::Jetro;
use serde_json::{json, Value};
use std::time::Instant;

use jaq_core::load::{Arena, File, Loader};
use jaq_core::{data, unwrap_valr, Compiler, Ctx, Vars};
use jaq_json::{read as jaq_read, Val as JaqVal};

const ITERS: usize = 10;

fn synth_doc(n_orders: usize, items_per_order: usize) -> Value {
    let regions = [
        "us-east",
        "us-west",
        "eu-central",
        "ap-southeast",
        "sa-south",
    ];
    let statuses = ["pending", "shipped", "delivered", "cancelled", "refunded"];
    let priorities = ["low", "normal", "high", "urgent"];
    let categories = [
        "electronics",
        "books",
        "apparel",
        "grocery",
        "toys",
        "tools",
    ];

    let mut orders = Vec::with_capacity(n_orders);
    for i in 0..n_orders {
        let mut items = Vec::with_capacity(items_per_order);
        let mut total: f64 = 0.0;
        for j in 0..items_per_order {
            let price = ((i * 7 + j * 13) % 500) as f64 + 9.99;
            let qty = ((i + j) % 5 + 1) as i64;
            total += price * qty as f64;
            items.push(json!({
                "sku": format!("SKU-{:05}", (i * items_per_order + j) % 9973),
                "name": format!("item-{}-{}", i, j),
                "category": categories[(i + j) % categories.len()],
                "price": price,
                "qty": qty,
            }));
        }
        orders.push(json!({
            "id": 100_000 + i,
            "status": statuses[i % statuses.len()],
            "priority": priorities[(i / 3) % priorities.len()],
            "region": regions[i % regions.len()],
            "total": (total * 100.0).round() / 100.0,
            "customer": {
                "id": 10_000 + (i % 5000),
                "name": format!("Customer {}", i % 5000),
                "email": format!("c{}@example.com", i % 5000),
                "address": {
                    "city": match i % 6 {
                        0 => "Tokyo", 1 => "Berlin", 2 => "São Paulo",
                        3 => "Nairobi", 4 => "Austin", _ => "Toronto"
                    },
                    "zip": format!("{:05}", (i * 17) % 100_000),
                    "country_code": match i % 6 {
                        0 => "JP", 1 => "DE", 2 => "BR",
                        3 => "KE", 4 => "US", _ => "CA"
                    },
                }
            },
            "items": items,
        }));
    }
    json!({ "orders": orders, "meta": { "kind": "bench_vs_jaq", "version": 1 } })
}

#[derive(Clone, Copy)]
struct Stats {
    best: u128,
    median: u128,
    mean: u128,
}

fn sample<F: FnMut()>(mut f: F) -> Stats {
    let _ = f(); // warmup
    let mut samples = Vec::with_capacity(ITERS);
    for _ in 0..ITERS {
        let t = Instant::now();
        f();
        samples.push(t.elapsed().as_micros());
    }
    samples.sort();
    let best = samples[0];
    let median = samples[samples.len() / 2];
    let mean = samples.iter().sum::<u128>() / samples.len() as u128;
    Stats { best, median, mean }
}

fn show(label: &str, s: Stats) {
    println!(
        "  {:<14} best {:>8}µs  median {:>8}µs  mean {:>8}µs",
        label, s.best, s.median, s.mean
    );
}

// ── jaq in-process runner ────────────────────────────────────────────────────

fn compile_jaq(
    code: &str,
) -> &'static jaq_core::compile::Filter<jaq_core::Native<data::JustLut<JaqVal>>> {
    let arena: &'static Arena = Box::leak(Box::new(Arena::default()));
    let defs = jaq_core::defs()
        .chain(jaq_std::defs())
        .chain(jaq_json::defs());
    let loader = Loader::new(defs);
    let modules = loader.load(arena, File { code, path: () }).unwrap();
    let funs = jaq_core::funs()
        .chain(jaq_std::funs())
        .chain(jaq_json::funs());
    let filter = Compiler::default()
        .with_funs(funs)
        .compile(modules)
        .unwrap();
    Box::leak(Box::new(filter))
}

fn run_jaq(
    filter: &'static jaq_core::compile::Filter<jaq_core::Native<data::JustLut<JaqVal>>>,
    input: &JaqVal,
) -> usize {
    let ctx = Ctx::<data::JustLut<JaqVal>>::new(&filter.lut, Vars::new([]));
    filter.id.run((ctx, input.clone())).map(unwrap_valr).count()
}

// ── Benchmark harness ────────────────────────────────────────────────────────

fn bench(
    label: &str,
    jetro_tree: &Jetro,
    jetro_scan: Option<&Jetro>,
    jetro_q: &str,
    jaq_q: &str,
    jaq_input: &JaqVal,
) {
    println!("\n{}", label);
    println!("  jetro: {}", jetro_q);
    println!("  jq   : {}", jaq_q);

    // `collect_val` keeps the result as jetro's native `Val` — parity with
    // jaq, which returns its own `Val` iterator without materialising to
    // `serde_json::Value`.  `collect` would add a deep Arc<str>→String
    // clone of every key + a full tree rebuild, which dominates structural
    // results like `group_by` on 20k items and unfairly penalises jetro.
    let t = sample(|| {
        let _ = jetro_tree.collect_val(jetro_q).unwrap();
    });
    show("jetro-tree", t);

    if let Some(js) = jetro_scan {
        let s = sample(|| {
            let _ = js.collect_val(jetro_q).unwrap();
        });
        show("jetro-scan", s);
    }

    let compiled = compile_jaq(jaq_q);
    let j = sample(|| {
        let _ = run_jaq(compiled, jaq_input);
    });
    show("jaq", j);

    let ratio = j.median as f64 / t.median.max(1) as f64;
    println!(
        "  jetro-tree median vs jaq: {:.2}x (jetro {} times faster)",
        ratio,
        if ratio >= 1.0 { "" } else { "slower —" }
    );
    if let Some(js) = jetro_scan {
        let s = sample(|| {
            let _ = js.collect_val(jetro_q).unwrap();
        });
        let r = j.median as f64 / s.median.max(1) as f64;
        println!("  jetro-scan median vs jaq: {:.2}x", r);
    }
}

fn main() {
    let n_orders = 20_000usize;
    let items_per_order = 6usize;
    let doc = synth_doc(n_orders, items_per_order);
    let bytes = serde_json::to_vec(&doc).unwrap();
    let mb = bytes.len() as f64 / 1_048_576.0;
    println!(
        "payload: {} orders × {} items = {} items, {:.2} MB, iters: {}",
        n_orders,
        items_per_order,
        n_orders * items_per_order,
        mb,
        ITERS
    );

    let j_tree = Jetro::new(doc.clone());
    let j_scan = Jetro::from_bytes(bytes.clone()).unwrap();

    // Build jaq input once from the JSON bytes.
    let jaq_input: JaqVal = jaq_read::parse_single(&bytes).unwrap();

    bench(
        "Q1  shallow field projection (shape-repeat)",
        &j_tree,
        None,
        "$.orders.map(customer.address.city)",
        "[.orders[] | .customer.address.city]",
        &jaq_input,
    );

    bench(
        "Q2  projection + unique",
        &j_tree,
        None,
        "$.orders.map(customer.address.country_code).unique()",
        "[.orders[].customer.address.country_code] | unique",
        &jaq_input,
    );

    bench(
        "Q3  filter + project",
        &j_tree,
        None,
        "$.orders.filter(total > 500).map(id)",
        "[.orders[] | select(.total > 500) | .id]",
        &jaq_input,
    );

    bench(
        "Q4  multi-condition filter + count",
        &j_tree,
        None,
        r#"$.orders.filter(status == "shipped" and priority == "high").count()"#,
        r#"[.orders[] | select(.status == "shipped" and .priority == "high")] | length"#,
        &jaq_input,
    );

    bench(
        "Q5  deep find: broad match",
        &j_tree,
        Some(&j_scan),
        r#"$..find(@.status == "shipped")"#,
        r#"[.. | objects | select(.status? == "shipped")]"#,
        &jaq_input,
    );

    bench(
        "Q6  deep find: narrow (1 hit)",
        &j_tree,
        Some(&j_scan),
        r#"$..find(@.sku == "SKU-00042")"#,
        r#"[.. | objects | select(.sku? == "SKU-00042")]"#,
        &jaq_input,
    );

    bench(
        "Q7  deep find: multi-predicate AND",
        &j_tree,
        Some(&j_scan),
        r#"$..find(@.status == "shipped", @.priority == "urgent")"#,
        r#"[.. | objects | select(.status? == "shipped" and .priority? == "urgent")]"#,
        &jaq_input,
    );

    bench(
        "Q8  deep key extract + aggregate",
        &j_tree,
        Some(&j_scan),
        "$..total.sum()",
        "[.. | objects | .total? // empty] | add",
        &jaq_input,
    );

    bench(
        "Q9  deep key extract (120k values)",
        &j_tree,
        Some(&j_scan),
        "$..sku",
        "[.. | .sku? // empty]",
        &jaq_input,
    );

    bench(
        "Q10 group_by(status)",
        &j_tree,
        None,
        "$.orders.group_by(status)",
        ".orders | group_by(.status)",
        &jaq_input,
    );

    bench(
        "Q11 map(total).sum()",
        &j_tree,
        None,
        "$.orders.map(total).sum()",
        "[.orders[].total] | add",
        &jaq_input,
    );

    bench(
        "Q12 max over mapped field",
        &j_tree,
        None,
        "$.orders.map(total).max()",
        "[.orders[].total] | max",
        &jaq_input,
    );

    bench(
        "Q13 list-comp equivalent",
        &j_tree,
        None,
        "[o.id for o in $.orders if o.total > 1000]",
        "[.orders[] | select(.total > 1000) | .id]",
        &jaq_input,
    );
}
