//! Bench `$..price.sum()` over a deep doc — tape vs Val.
//! cargo run --release --example bench_tape_descend --features simd-json -p jetro-core
use jetro_core::Jetro;
use std::time::Instant;

fn make_doc(n: usize) -> Vec<u8> {
    let mut s = String::with_capacity(n * 100);
    s.push_str("{\"orders\":[");
    for i in 0..n {
        if i > 0 { s.push(','); }
        s.push_str(&format!(
            "{{\"id\":{},\"items\":[{{\"price\":{}}},{{\"price\":{}}}]}}",
            i, i * 2, i * 3));
    }
    s.push_str("]}");
    s.into_bytes()
}

fn bench<F: FnMut() -> u64>(name: &str, iters: usize, mut f: F) {
    let _ = f();
    let t = Instant::now();
    let mut acc: u64 = 0;
    for _ in 0..iters { acc = acc.wrapping_add(f()); }
    let per = t.elapsed() / iters as u32;
    println!("{:<40} {:>9.2} µs / iter   (acc {})", name, per.as_secs_f64() * 1_000_000.0, acc);
}

fn main() {
    let n: usize = std::env::var("N").ok().and_then(|s| s.parse().ok()).unwrap_or(20_000);
    let iters: usize = std::env::var("ITERS").ok().and_then(|s| s.parse().ok()).unwrap_or(50);
    let bytes = make_doc(n);
    println!("doc = {} orders, {:.2} MB; iters = {}\n", n, bytes.len() as f64 / 1_000_000.0, iters);

    // from_slice (Val tree) vs from_simd_lazy (tape).
    let j_val  = Jetro::from_slice(&bytes).unwrap();
    let j_tape = Jetro::from_simd_lazy(bytes.clone()).unwrap();
    // Warm both root_val + tape compile cache.
    let _ = j_val.collect("$..price.sum()").unwrap();
    let _ = j_tape.collect("$..price.sum()").unwrap();

    bench("Val tree     $..price.sum()", iters, || {
        let r = j_val.collect("$..price.sum()").unwrap();
        r.as_i64().unwrap_or(0) as u64
    });
    bench("tape walker  $..price.sum()", iters, || {
        let r = j_tape.collect("$..price.sum()").unwrap();
        r.as_i64().unwrap_or(0) as u64
    });
    bench("Val tree     $..price.count()", iters, || {
        let r = j_val.collect("$..price.count()").unwrap();
        r.as_i64().unwrap_or(0) as u64
    });
    bench("tape walker  $..price.count()", iters, || {
        let r = j_tape.collect("$..price.count()").unwrap();
        r.as_i64().unwrap_or(0) as u64
    });
    bench("Val tree     $..price (bare)", iters, || {
        let r = j_val.collect("$..price").unwrap();
        r.as_array().map(|a| a.len() as u64).unwrap_or(0)
    });
    bench("tape walker  $..price (bare)", iters, || {
        let r = j_tape.collect("$..price").unwrap();
        r.as_array().map(|a| a.len() as u64).unwrap_or(0)
    });

    // Cold-start: parse + first query.  Val tree path pays Val::from_json_simd
    // up-front; tape lazy path pays only TapeData::parse and walks tape.
    bench("cold from_slice + sum         ", iters, || {
        let j = Jetro::from_slice(&bytes).unwrap();
        let r = j.collect("$..price.sum()").unwrap();
        r.as_i64().unwrap_or(0) as u64
    });
    bench("cold from_simd_lazy + sum     ", iters, || {
        let j = Jetro::from_simd_lazy(bytes.clone()).unwrap();
        let r = j.collect("$..price.sum()").unwrap();
        r.as_i64().unwrap_or(0) as u64
    });

    // $.<arr>.map(<field>).<agg>() — Q12 bench_complex shape, hottest
    // typed-struct-comparable workload.  Need a doc with a top-level array
    // of homogeneous objects.
    let mut s = String::with_capacity(n * 60);
    s.push_str("{\"orders\":[");
    for i in 0..n {
        if i > 0 { s.push(','); }
        s.push_str(&format!("{{\"id\":{},\"total\":{}}}", i, i));
    }
    s.push_str("]}");
    let arr_bytes = s.into_bytes();

    let j_val_arr  = Jetro::from_slice(&arr_bytes).unwrap();
    let j_tape_arr = Jetro::from_simd_lazy(arr_bytes.clone()).unwrap();
    let _ = j_val_arr.collect("$.orders.map(total).sum()").unwrap();
    let _ = j_tape_arr.collect("$.orders.map(total).sum()").unwrap();

    bench("Val tree     $.orders.map(total).sum()", iters, || {
        let r = j_val_arr.collect("$.orders.map(total).sum()").unwrap();
        r.as_i64().unwrap_or(0) as u64
    });
    bench("tape walker  $.orders.map(total).sum()", iters, || {
        let r = j_tape_arr.collect("$.orders.map(total).sum()").unwrap();
        r.as_i64().unwrap_or(0) as u64
    });
    bench("Val tree     $.orders.map(total).max()", iters, || {
        let r = j_val_arr.collect("$.orders.map(total).max()").unwrap();
        r.as_f64().unwrap_or(0.0) as u64
    });
    bench("tape walker  $.orders.map(total).max()", iters, || {
        let r = j_tape_arr.collect("$.orders.map(total).max()").unwrap();
        r.as_f64().unwrap_or(0.0) as u64
    });

    // filter+count tape path
    let _ = j_val_arr.collect("$.orders.filter(total > 100).count()").unwrap();
    let _ = j_tape_arr.collect("$.orders.filter(total > 100).count()").unwrap();
    bench("Val tree     $.orders.filter(total>100).count()", iters, || {
        let r = j_val_arr.collect("$.orders.filter(total > 100).count()").unwrap();
        r.as_i64().unwrap_or(0) as u64
    });
    bench("tape walker  $.orders.filter(total>100).count()", iters, || {
        let r = j_tape_arr.collect("$.orders.filter(total > 100).count()").unwrap();
        r.as_i64().unwrap_or(0) as u64
    });

    let _ = j_val_arr.collect("$.orders.filter(total > 100).map(total).sum()").unwrap();
    let _ = j_tape_arr.collect("$.orders.filter(total > 100).map(total).sum()").unwrap();
    bench("Val tree     $.orders.filter(total>100).map(total).sum()", iters, || {
        let r = j_val_arr.collect("$.orders.filter(total > 100).map(total).sum()").unwrap();
        r.as_i64().unwrap_or(0) as u64
    });
    bench("tape walker  $.orders.filter(total>100).map(total).sum()", iters, || {
        let r = j_tape_arr.collect("$.orders.filter(total > 100).map(total).sum()").unwrap();
        r.as_i64().unwrap_or(0) as u64
    });
}
