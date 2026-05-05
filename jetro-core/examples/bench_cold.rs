// Cold single-run bench: native Rust vs jaq vs jetro.
// Each engine starts fresh per query: parse + compile + execute + serialize.
// No warmup. No iterations. Measures total startup + eval cost.
//
// Queries are chains of demanding builtins on N=8_000 records:
//   filter x 2, sort, skip, take, flat_map, map, sum, avg, len, unique,
//   count_by / group_by, fstring interpolation.

use std::time::Instant;

use jaq_core::{
    data,
    load::{Arena, File, Loader},
    Compiler, Ctx, Vars,
};
use jaq_json::Val as JaqVal;
use serde::{Deserialize, Serialize};

#[derive(Deserialize)]
struct Record<'a> {
    id: i64,
    #[serde(borrow)]
    user: User<'a>,
    items: Vec<Item<'a>>,
    #[serde(borrow)]
    tags: Vec<&'a str>,
    active: bool,
    score: i64,
}

#[derive(Deserialize)]
struct User<'a> {
    #[serde(borrow)]
    name: &'a str,
    #[allow(dead_code)]
    age: i64,
    addr: Addr<'a>,
}

#[derive(Deserialize)]
struct Addr<'a> {
    #[serde(borrow)]
    city: &'a str,
    #[serde(borrow)]
    zip: &'a str,
}

#[derive(Deserialize)]
struct Item<'a> {
    #[serde(borrow)]
    sku: &'a str,
    qty: i64,
    price: f64,
}

#[derive(Deserialize)]
struct Wrapper<'a> {
    #[serde(borrow)]
    data: Vec<Record<'a>>,
}

fn build_doc(n: usize) -> String {
    let mut s = String::from(r#"{"data":["#);
    let cities = [
        "NYC", "SF", "LA", "Boston", "Seattle", "Austin", "Miami", "Chicago",
    ];
    for i in 0..n {
        if i > 0 {
            s.push(',');
        }
        let city = cities[i % cities.len()];
        let mut items = String::from("[");
        let item_count = 3 + (i % 5);
        for k in 0..item_count {
            if k > 0 {
                items.push(',');
            }
            items.push_str(&format!(
                r#"{{"sku":"S{i}_{k}","qty":{},"price":{}}}"#,
                (k + 1) * 2,
                ((i * 7 + k * 13) % 200) as f64 + 9.99
            ));
        }
        items.push(']');
        s.push_str(&format!(
            r#"{{"id":{i},"user":{{"name":"user_{i}","age":{},"addr":{{"city":"{city}","zip":"{}"}}}},"items":{items},"tags":["t{}","t{}","t{}"],"active":{},"score":{}}}"#,
            18 + (i % 60),
            10000 + (i % 1000),
            i % 5,
            (i + 1) % 5,
            (i + 2) % 5,
            if i % 3 == 0 { "true" } else { "false" },
            (i * 37) % 1000,
        ));
    }
    s.push_str("]}");
    s
}

fn time_one<F: FnOnce()>(f: F) -> f64 {
    let t0 = Instant::now();
    f();
    t0.elapsed().as_secs_f64() * 1000.0
}

fn run_jaq_cold(code: &str, jaq_input: &JaqVal) {
    let program = File { code, path: () };
    let defs = jaq_core::defs()
        .chain(jaq_std::defs())
        .chain(jaq_json::defs());
    let funs = jaq_core::funs()
        .chain(jaq_std::funs())
        .chain(jaq_json::funs());
    let loader = Loader::new(defs);
    let arena = Arena::default();
    let modules = loader.load(&arena, program).expect("parse");
    let filter = Compiler::default()
        .with_funs(funs)
        .compile(modules)
        .expect("compile");
    let ctx = Ctx::<data::JustLut<JaqVal>>::new(&filter.lut, Vars::new([]));
    let pp = jaq_json::write::Pp::<String>::default();
    let out: Vec<_> = filter.id.run((ctx, jaq_input.clone())).collect();
    let mut buf: Vec<u8> = Vec::new();
    for r in &out {
        let v = r.clone().expect("run jq");
        jaq_json::write::write(&mut buf, &pp, 0, &v).unwrap();
    }
    std::hint::black_box(buf);
}

fn run_jetro_cold(code: &str, doc_bytes: &[u8]) -> Option<()> {
    let jetro = jetro_core::Jetro::from_bytes(doc_bytes.to_vec()).ok()?;
    let v = jetro.collect(code).ok()?;
    std::hint::black_box(serde_json::to_vec(&v).unwrap());
    Some(())
}

fn print_row(name: &str, native: f64, jetro: f64, jaq: f64) {
    let r = |a: f64, b: f64| format!("{:.2}x", b / a);
    println!(
        "{name:<44} native {native:>7.3}ms  jetro {jetro:>7.3}ms ({})  jaq {jaq:>7.3}ms ({})",
        r(native, jetro),
        r(native, jaq),
    );
}

fn main() {
    let n: usize = 8_000;
    let json = build_doc(n);
    println!(
        "Cold single-run bench - N={n}  input={} KB  (no warmup, no iters)\n",
        json.len() / 1024
    );

    let wrapper: Wrapper = serde_json::from_str(&json).unwrap();
    let data: &Vec<Record> = &wrapper.data;
    let jaq_input = jaq_json::read::parse_single(json.as_bytes()).unwrap();
    let doc = json.as_bytes();

    // 1. filter(active) -> filter(score>200) -> sort(-score) -> take(100)
    //    -> flat_map(items) -> filter(price>50) -> map(qty*price) -> sum
    {
        let native = time_one(|| {
            let mut active: Vec<&Record> =
                data.iter().filter(|r| r.active && r.score > 200).collect();
            active.sort_unstable_by(|a, b| b.score.cmp(&a.score));
            let total: f64 = active
                .iter()
                .take(100)
                .flat_map(|r| r.items.iter())
                .filter(|it| it.price > 50.0)
                .map(|it| it.qty as f64 * it.price)
                .sum();
            std::hint::black_box(serde_json::to_vec(&total).unwrap());
        });
        let jq = "$.data.filter(active).filter(score > 200).sort(-score).take(100).flat_map(items).filter(price > 50).map(qty * price).sum()";
        let jetro = time_one(|| {
            let _ = run_jetro_cold(jq, doc);
        });
        let q = "[.data[] | select(.active) | select(.score > 200)] | sort_by(-.score) | .[:100] | [.[].items[] | select(.price > 50) | .qty * .price] | add";
        let jaq = time_one(|| run_jaq_cold(q, &jaq_input));
        print_row("active top-100 expensive-item revenue", native, jetro, jaq);
    }

    // 2. flat_map(items) -> sort(-price) -> take(30) -> map({sku,price})
    //    sorts the entire item corpus (~35k items at N=8000)
    {
        let native = time_one(|| {
            #[derive(Serialize)]
            struct Out<'a> {
                sku: &'a str,
                price: f64,
            }
            let mut all: Vec<(&str, f64)> = data
                .iter()
                .flat_map(|r| r.items.iter().map(|it| (it.sku, it.price)))
                .collect();
            all.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            let out: Vec<Out> = all
                .iter()
                .take(30)
                .map(|&(s, p)| Out { sku: s, price: p })
                .collect();
            std::hint::black_box(serde_json::to_vec(&out).unwrap());
        });
        let jq = "$.data.flat_map(items).sort(-price).take(30).map({sku, price})";
        let jetro = time_one(|| {
            let _ = run_jetro_cold(jq, doc);
        });
        let q = "[.data[].items[]] | sort_by(-.price) | .[:30] | map({sku, price})";
        let jaq = time_one(|| run_jaq_cold(q, &jaq_input));
        print_row("flatmap+sort all-items+take+project", native, jetro, jaq);
    }

    // 3. sort(-score) -> skip(200) -> take(50) -> map({id, city, score})
    {
        let native = time_one(|| {
            #[derive(Serialize)]
            struct Out<'a> {
                id: i64,
                city: &'a str,
                score: i64,
            }
            let mut sorted: Vec<&Record> = data.iter().collect();
            sorted.sort_unstable_by(|a, b| b.score.cmp(&a.score));
            let out: Vec<Out> = sorted
                .iter()
                .skip(200)
                .take(50)
                .map(|r| Out {
                    id: r.id,
                    city: r.user.addr.city,
                    score: r.score,
                })
                .collect();
            std::hint::black_box(serde_json::to_vec(&out).unwrap());
        });
        let jq = "$.data.sort(-score).skip(200).take(50).map({id, city: user.addr.city, score})";
        let jetro = time_one(|| {
            let _ = run_jetro_cold(jq, doc);
        });
        let q = ".data | sort_by(-.score) | .[200:250] | map({id, city: .user.addr.city, score})";
        let jaq = time_one(|| run_jaq_cold(q, &jaq_input));
        print_row("sort+skip+take+project", native, jetro, jaq);
    }

    // 4. filter(active) -> flat_map(tags) -> unique
    //    blows up to ~N/3 x 3 = ~8000 strings before dedup
    {
        let native = time_one(|| {
            let mut tags: Vec<&str> = data
                .iter()
                .filter(|r| r.active)
                .flat_map(|r| r.tags.iter().copied())
                .collect();
            tags.sort_unstable();
            tags.dedup();
            std::hint::black_box(serde_json::to_vec(&tags).unwrap());
        });
        let jq = "$.data.filter(active).flat_map(tags).unique()";
        let jetro = time_one(|| {
            let _ = run_jetro_cold(jq, doc);
        });
        let q = "[.data[] | select(.active) | .tags[]] | unique";
        let jaq = time_one(|| run_jaq_cold(q, &jaq_input));
        print_row("filter+flatmap-tags+unique", native, jetro, jaq);
    }

    // 5. flat_map(items) -> filter(price>100) -> map(qty*price) -> sum
    {
        let native = time_one(|| {
            let total: f64 = data
                .iter()
                .flat_map(|r| r.items.iter())
                .filter(|it| it.price > 100.0)
                .map(|it| it.qty as f64 * it.price)
                .sum();
            std::hint::black_box(serde_json::to_vec(&total).unwrap());
        });
        let jq = "$.data.flat_map(items).filter(price > 100).map(qty * price).sum()";
        let jetro = time_one(|| {
            let _ = run_jetro_cold(jq, doc);
        });
        let q = "[.data[].items[] | select(.price > 100) | .qty * .price] | add";
        let jaq = time_one(|| run_jaq_cold(q, &jaq_input));
        print_row("flatmap+filter+map-arith+sum", native, jetro, jaq);
    }

    // 6. filter(active) -> sort(-score) -> take(50) -> map(fstring)
    {
        let native = time_one(|| {
            let mut active: Vec<&Record> = data.iter().filter(|r| r.active).collect();
            active.sort_unstable_by(|a, b| b.score.cmp(&a.score));
            let out: Vec<String> = active
                .iter()
                .take(50)
                .map(|r| {
                    format!(
                        "#{} {} ({}) score={}",
                        r.id, r.user.name, r.user.addr.city, r.score
                    )
                })
                .collect();
            std::hint::black_box(serde_json::to_vec(&out).unwrap());
        });
        let jq = r##"$.data.filter(active).sort(-score).take(50).map(f"#{id} {user.name} ({user.addr.city}) score={score}")"##;
        let jetro = time_one(|| {
            let _ = run_jetro_cold(jq, doc);
        });
        let q = r##"[.data[] | select(.active)] | sort_by(-.score) | .[:50] | [.[] | "#\(.id) \(.user.name) (\(.user.addr.city)) score=\(.score)"]"##;
        let jaq = time_one(|| run_jaq_cold(q, &jaq_input));
        print_row("filter+sort+take+fstring", native, jetro, jaq);
    }

    // 7. filter(score>700) -> flat_map(items) -> map(price) -> avg
    //    large fanout: ~N/10 records x ~5 items each
    {
        let native = time_one(|| {
            let prices: Vec<f64> = data
                .iter()
                .filter(|r| r.score > 700)
                .flat_map(|r| r.items.iter().map(|it| it.price))
                .collect();
            let avg = if prices.is_empty() {
                0.0
            } else {
                prices.iter().sum::<f64>() / prices.len() as f64
            };
            std::hint::black_box(serde_json::to_vec(&avg).unwrap());
        });
        let jq = "$.data.filter(score > 700).flat_map(items).map(price).avg()";
        let jetro = time_one(|| {
            let _ = run_jetro_cold(jq, doc);
        });
        let q = "[.data[] | select(.score > 700) | .items[].price] | (add / length)";
        let jaq = time_one(|| run_jaq_cold(q, &jaq_input));
        print_row("filter+flatmap+avg", native, jetro, jaq);
    }

    // 8. sort(-score) -> take(20) -> map({id, city, total: items.sum(qty*price)})
    //    nested arithmetic inside map projection
    {
        let native = time_one(|| {
            #[derive(Serialize)]
            struct Out<'a> {
                id: i64,
                city: &'a str,
                total: f64,
            }
            let mut sorted: Vec<&Record> = data.iter().collect();
            sorted.sort_unstable_by(|a, b| b.score.cmp(&a.score));
            let out: Vec<Out> = sorted
                .iter()
                .take(20)
                .map(|r| Out {
                    id: r.id,
                    city: r.user.addr.city,
                    total: r.items.iter().map(|it| it.qty as f64 * it.price).sum(),
                })
                .collect();
            std::hint::black_box(serde_json::to_vec(&out).unwrap());
        });
        let jq = "$.data.sort(-score).take(20).map({id, city: user.addr.city, total: items.map(qty * price).sum()})";
        let jetro = time_one(|| {
            let _ = run_jetro_cold(jq, doc);
        });
        let q = ".data | sort_by(-.score) | .[:20] | map({id, city: .user.addr.city, total: ([.items[] | .qty * .price] | add)})";
        let jaq = time_one(|| run_jaq_cold(q, &jaq_input));
        print_row("sort+take+nested-computed-projection", native, jetro, jaq);
    }

    // 9. filter(active) -> filter(score>500) -> flat_map(items)
    //    -> filter(price>75) -> filter(qty>2) -> len
    //    five-stage chain with two filter passes per item
    {
        let native = time_one(|| {
            let count: usize = data
                .iter()
                .filter(|r| r.active && r.score > 500)
                .flat_map(|r| r.items.iter())
                .filter(|it| it.price > 75.0 && it.qty > 2)
                .count();
            std::hint::black_box(serde_json::to_vec(&count).unwrap());
        });
        let jq = "$.data.filter(active).filter(score > 500).flat_map(items).filter(price > 75).filter(qty > 2).len()";
        let jetro = time_one(|| {
            let _ = run_jetro_cold(jq, doc);
        });
        let q = "[.data[] | select(.active) | select(.score > 500) | .items[] | select(.price > 75) | select(.qty > 2)] | length";
        let jaq = time_one(|| run_jaq_cold(q, &jaq_input));
        print_row("5-stage filter chain + count", native, jetro, jaq);
    }

    // 10. count_by(active) - group all records by active flag, emit {true: N, false: M}
    //     jetro: count_by(active)  |  jaq: group_by(.active) -> map -> sort
    {
        let native = time_one(|| {
            let (t, f) =
                data.iter().fold(
                    (0u64, 0u64),
                    |(t, f), r| {
                        if r.active {
                            (t + 1, f)
                        } else {
                            (t, f + 1)
                        }
                    },
                );
            std::hint::black_box(
                serde_json::to_vec(&serde_json::json!({"true": t, "false": f})).unwrap(),
            );
        });
        let jq = "$.data.count_by(active)";
        let jetro = time_one(|| {
            let _ = run_jetro_cold(jq, doc);
        });
        let q = ".data | group_by(.active) | map({(.[0].active | tostring): length}) | add";
        let jaq = time_one(|| run_jaq_cold(q, &jaq_input));
        print_row("count_by(active) / group_by+map", native, jetro, jaq);
    }

    // 11. sort(-score) -> take(300) -> map(user.addr.zip) -> unique
    //     unique over a large projected array
    {
        let native = time_one(|| {
            let mut sorted: Vec<&Record> = data.iter().collect();
            sorted.sort_unstable_by(|a, b| b.score.cmp(&a.score));
            let mut zips: Vec<&str> = sorted.iter().take(300).map(|r| r.user.addr.zip).collect();
            zips.sort_unstable();
            zips.dedup();
            std::hint::black_box(serde_json::to_vec(&zips).unwrap());
        });
        let jq = "$.data.sort(-score).take(300).map(user.addr.zip).unique()";
        let jetro = time_one(|| {
            let _ = run_jetro_cold(jq, doc);
        });
        let q = ".data | sort_by(-.score) | .[:300] | [.[].user.addr.zip] | unique";
        let jaq = time_one(|| run_jaq_cold(q, &jaq_input));
        print_row("sort+take+map+unique (top-300 zips)", native, jetro, jaq);
    }

    // 12. flat_map(items) -> sort(-price) -> unique (by price value) -> len
    //     unique on large float array - forces sort + dedup pass
    {
        let native = time_one(|| {
            let mut prices: Vec<i64> = data
                .iter()
                .flat_map(|r| r.items.iter().map(|it| (it.price * 100.0) as i64))
                .collect();
            prices.sort_unstable();
            prices.dedup();
            std::hint::black_box(serde_json::to_vec(&prices.len()).unwrap());
        });
        let jq = "$.data.flat_map(items).map(price).unique().len()";
        let jetro = time_one(|| {
            let _ = run_jetro_cold(jq, doc);
        });
        let q = "[.data[].items[].price] | unique | length";
        let jaq = time_one(|| run_jaq_cold(q, &jaq_input));
        print_row("flatmap+map+unique+len (all prices)", native, jetro, jaq);
    }

    println!("\n--- v2 cases ---\n");

    // 13. filter+map+sum
    {
        let native = time_one(|| {
            let s: i64 = data.iter().filter(|r| r.active).map(|r| r.score).sum();
            std::hint::black_box(serde_json::to_vec(&s).unwrap());
        });
        let jetro = time_one(|| {
            let _ = run_jetro_cold("$.data.filter(active).map(score).sum()", doc);
        });
        let jaq =
            time_one(|| run_jaq_cold("[.data[] | select(.active) | .score] | add", &jaq_input));
        print_row("filter+map+sum", native, jetro, jaq);
    }

    // 14. flat_map+filter+count
    {
        let native = time_one(|| {
            let n: usize = data
                .iter()
                .flat_map(|r| r.items.iter())
                .filter(|it| it.price > 50.0)
                .count();
            std::hint::black_box(serde_json::to_vec(&n).unwrap());
        });
        let jetro = time_one(|| {
            let _ = run_jetro_cold("$.data.flat_map(items).filter(price > 50).len()", doc);
        });
        let jaq = time_one(|| {
            run_jaq_cold(
                "[.data[] | .items[] | select(.price > 50)] | length",
                &jaq_input,
            )
        });
        print_row("flat_map+filter+count", native, jetro, jaq);
    }

    // 15. filter+flat_map+map+sum
    {
        let native = time_one(|| {
            let s: f64 = data
                .iter()
                .filter(|r| r.active)
                .flat_map(|r| r.items.iter().map(|it| it.qty as f64 * it.price))
                .sum();
            std::hint::black_box(serde_json::to_vec(&s).unwrap());
        });
        let jetro = time_one(|| {
            let _ = run_jetro_cold(
                "$.data.filter(active).flat_map(items).map(qty * price).sum()",
                doc,
            );
        });
        let jaq = time_one(|| {
            run_jaq_cold(
                "[.data[] | select(.active) | .items[] | .qty * .price] | add",
                &jaq_input,
            )
        });
        print_row("filter+flat_map+map+sum", native, jetro, jaq);
    }

    // 16. sort_by+take+map (top10)
    {
        let native = time_one(|| {
            let mut idx: Vec<&Record> = data.iter().collect();
            idx.sort_by(|a, b| b.score.cmp(&a.score));
            #[derive(Serialize)]
            struct Top<'a> {
                id: i64,
                name: &'a str,
                score: i64,
            }
            let out: Vec<Top> = idx
                .iter()
                .take(10)
                .map(|r| Top {
                    id: r.id,
                    name: r.user.name,
                    score: r.score,
                })
                .collect();
            std::hint::black_box(serde_json::to_vec(&out).unwrap());
        });
        let jetro = time_one(|| {
            let _ = run_jetro_cold(
                "$.data.sort_by(-score).take(10).map({id, name: user.name, score})",
                doc,
            );
        });
        let jaq = time_one(|| {
            run_jaq_cold(
                ".data | sort_by(-.score) | .[:10] | map({id, name: .user.name, score})",
                &jaq_input,
            )
        });
        print_row("sort_by+take+map (top10)", native, jetro, jaq);
    }

    // 17. map+unique cities
    {
        let native = time_one(|| {
            let mut seen: std::collections::BTreeSet<&str> = Default::default();
            for r in data {
                seen.insert(r.user.addr.city);
            }
            let out: Vec<&str> = seen.into_iter().collect();
            std::hint::black_box(serde_json::to_vec(&out).unwrap());
        });
        let jetro = time_one(|| {
            let _ = run_jetro_cold("$.data.map(user.addr.city).unique()", doc);
        });
        let jaq = time_one(|| run_jaq_cold("[.data[] | .user.addr.city] | unique", &jaq_input));
        print_row("map+unique (cities)", native, jetro, jaq);
    }

    // 18. deep projection
    {
        let native = time_one(|| {
            #[derive(Serialize)]
            struct DeepOut<'a> {
                id: i64,
                city: &'a str,
                item_count: usize,
                total: f64,
            }
            let out: Vec<DeepOut> = data
                .iter()
                .map(|r| DeepOut {
                    id: r.id,
                    city: r.user.addr.city,
                    item_count: r.items.len(),
                    total: r.items.iter().map(|it| it.qty as f64 * it.price).sum(),
                })
                .collect();
            std::hint::black_box(serde_json::to_vec(&out).unwrap());
        });
        let jetro = time_one(|| {
            let _ = run_jetro_cold(
                "$.data.map({id, city: user.addr.city, item_count: items.len(), total: items.map(qty * price).sum()})",
                doc,
            );
        });
        let jaq = time_one(|| {
            run_jaq_cold(
                "[.data[] | {id, city: .user.addr.city, item_count: (.items | length), total: ([.items[] | .qty * .price] | add)}]",
                &jaq_input,
            )
        });
        print_row("map (deep projection)", native, jetro, jaq);
    }

    // 19. f-string per row
    {
        let native = time_one(|| {
            let out: Vec<String> = data
                .iter()
                .map(|r| {
                    format!(
                        "#{} {} ({}) ${}",
                        r.id, r.user.name, r.user.addr.city, r.score
                    )
                })
                .collect();
            std::hint::black_box(serde_json::to_vec(&out).unwrap());
        });
        let jetro = time_one(|| {
            let _ = run_jetro_cold(
                r##"$.data.map(f"#{id} {user.name} ({user.addr.city}) ${score}")"##,
                doc,
            );
        });
        let jaq = time_one(|| {
            run_jaq_cold(
                r##"[.data[] | "#\(.id) \(.user.name) (\(.user.addr.city)) $\(.score)"]"##,
                &jaq_input,
            )
        });
        print_row("map f-string", native, jetro, jaq);
    }

    // 20. flat_map+map all prices
    {
        let native = time_one(|| {
            let prices: Vec<f64> = data
                .iter()
                .flat_map(|r| r.items.iter().map(|it| it.price))
                .collect();
            std::hint::black_box(serde_json::to_vec(&prices).unwrap());
        });
        let jetro = time_one(|| {
            let _ = run_jetro_cold("$.data.flat_map(items).map(price)", doc);
        });
        let jaq = time_one(|| run_jaq_cold("[.data[] | .items[] | .price]", &jaq_input));
        print_row("flat_map+map (all prices)", native, jetro, jaq);
    }

    // 21. filter+first
    {
        let native = time_one(|| {
            let r = data.iter().find(|r| r.score > 900);
            std::hint::black_box(serde_json::to_vec(&r.map(|r| r.id)).unwrap());
        });
        let jetro = time_one(|| {
            let _ = run_jetro_cold("$.data.filter(score > 900).first()", doc);
        });
        let jaq = time_one(|| run_jaq_cold("first(.data[] | select(.score > 900))", &jaq_input));
        print_row("filter+first", native, jetro, jaq);
    }

    // 22. skip+take+map pagination
    {
        let native = time_one(|| {
            let out: Vec<&Record> = data.iter().skip(100).take(20).collect();
            #[derive(Serialize)]
            struct Page {
                id: i64,
            }
            let p: Vec<Page> = out.iter().map(|r| Page { id: r.id }).collect();
            std::hint::black_box(serde_json::to_vec(&p).unwrap());
        });
        let jetro = time_one(|| {
            let _ = run_jetro_cold("$.data.skip(100).take(20).map({id})", doc);
        });
        let jaq = time_one(|| run_jaq_cold(".data | .[100:120] | map({id})", &jaq_input));
        print_row("skip+take+map (pagination)", native, jetro, jaq);
    }

    // 23. filter+map+avg
    {
        let native = time_one(|| {
            let avg: f64 = {
                let scores: Vec<i64> = data.iter().filter(|r| r.active).map(|r| r.score).collect();
                if scores.is_empty() {
                    0.0
                } else {
                    scores.iter().sum::<i64>() as f64 / scores.len() as f64
                }
            };
            std::hint::black_box(serde_json::to_vec(&avg).unwrap());
        });
        let jetro = time_one(|| {
            let _ = run_jetro_cold("$.data.filter(active).map(score).avg()", doc);
        });
        let jaq = time_one(|| {
            run_jaq_cold(
                "[.data[] | select(.active) | .score] | add / length",
                &jaq_input,
            )
        });
        print_row("filter+map+avg", native, jetro, jaq);
    }
}
