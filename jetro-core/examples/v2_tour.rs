//! v2.2 feature tour — Tier 1 search, chain-style writes, and the
//! Python-style ternary.  Run with:
//!
//!     cargo run --release --example v2_tour -p jetro-core
//!
//! Every snippet is compiled through the same VM the Jetro facade uses,
//! so the compile + pointer caches hit across calls on the same doc.

use jetro_core::Jetro;
use serde_json::json;

fn show(label: &str, v: &serde_json::Value) {
    println!("── {label}");
    println!("{}\n", serde_json::to_string_pretty(v).unwrap());
}

fn main() {
    let doc = json!({
        "store": {
            "currency": "USD",
            "books": [
                {"id": "b1", "title": "Dune",        "author": "Frank Herbert",  "price": 12.99, "year": 1965, "tags": ["sci-fi","classic"],  "stock": 14, "ratings": [5,5,4,5,3]},
                {"id": "b2", "title": "Foundation",  "author": "Isaac Asimov",   "price":  9.99, "year": 1951, "tags": ["sci-fi","classic"],  "stock":  0, "ratings": [5,5,5]},
                {"id": "b3", "title": "Neuromancer", "author": "William Gibson", "price": 19.50, "year": 1984, "tags": ["cyberpunk"],         "stock":  3, "ratings": [4,5,4,5]},
                {"id": "b4", "title": "The Hobbit",  "author": "J.R.R. Tolkien", "price": 14.25, "year": 1937, "tags": ["fantasy","classic"], "stock": 22, "ratings": [5,5,5,5,4,5]},
                {"id": "b5", "title": "Hyperion",    "author": "Dan Simmons",    "price": 18.00, "year": 1989, "tags": ["sci-fi"],            "stock":  7, "ratings": [5,5,4]}
            ],
            "orders": [
                {"id": "o1", "status": "paid",    "total": 38.23},
                {"id": "o2", "status": "pending", "total": 19.50},
                {"id": "o3", "status": "paid",    "total": 54.00}
            ]
        }
    });

    let j = Jetro::from_bytes(serde_json::to_vec(&doc).unwrap()).unwrap();

    // ── Shallow search (Tier 1) ──────────────────────────────────────────────

    show(
        "find — first match (alias of filter + [0])",
        &j.collect(r#"$.store.books.find(title == "Dune")"#).unwrap(),
    );

    show(
        "find_all — alias of filter",
        &j.collect(r#"$.store.books.find_all(tags.includes("classic"))"#)
            .unwrap(),
    );

    show(
        "pick — project + rename (alias: src)",
        &j.collect("$.store.books.pick(title, slug: id, year)")
            .unwrap(),
    );

    show(
        "unique_by — dedup by derived key",
        &j.collect("$.store.books.sort(year).unique_by(author)")
            .unwrap(),
    );

    show(
        "collect — scalar → [scalar], arr → arr",
        &j.collect("$.store.books[0].tags.collect()").unwrap(),
    );

    // ── Deep search ──────────────────────────────────────────────────────────

    show(
        "$..find — every descendant satisfying pred",
        &j.collect("$..find(@ kind number and @ < 10)").unwrap(),
    );

    show(
        "$..shape — every object with all listed keys",
        &j.collect("$..shape({id, title, price})").unwrap(),
    );

    show(
        "$..like — every object with listed key==lit",
        &j.collect(r#"$..like({status: "paid"})"#).unwrap(),
    );

    // ── Chain-style writes (desugar into `patch` blocks) ─────────────────────

    show(
        ".set — replace a single leaf (returns full doc)",
        &j.collect("$.store.currency.set(\"EUR\")").unwrap(),
    );

    show(
        ".modify — rewrite a leaf using @",
        &j.collect("$.store.books[0].price.modify(@ * 0.9)").unwrap(),
    );

    show(
        ".unset — drop a child of the leaf object",
        &j.collect("$.store.books[0].unset(ratings)").unwrap(),
    );

    // ── Python-style ternary ─────────────────────────────────────────────────

    show(
        "ternary — chained right-assoc",
        &j.collect(
            r#"
            $.store.books.map(
                {title,
                 status: "out" if stock == 0
                         else "low" if stock < 5
                         else "ok"}
            )
        "#,
        )
        .unwrap(),
    );

    show(
        "ternary — const-fold (else branch never compiles)",
        &j.collect("42 if true else 0/0").unwrap(),
    );

    // ── Combined: search + conditional inside projection ─────────────────────

    show(
        "shape via pick + ternary — analyst view",
        &j.collect(
            r#"
            $.store.books
              .find_all(price >= 10)
              .pick(id, label: title, year)
              .map({
                  id,
                  label,
                  era: "classic" if year < 1970 else "modern"
              })
        "#,
        )
        .unwrap(),
    );
}
