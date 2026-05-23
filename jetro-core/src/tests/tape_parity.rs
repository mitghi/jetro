//! Parity coverage between the byte/tape engine and the VM fallback for
//! demand-sensitive pipeline shapes.

use super::common::vm_query;
use serde_json::{json, Value};

fn assert_tape_vm_eq(query: &str, doc: &Value) {
    let expected = vm_query(query, doc).unwrap_or_else(|err| panic!("VM failed for {query}: {err}"));
    let bytes = serde_json::to_vec(doc).expect("serialize fixture");
    let jetro = crate::Jetro::from_bytes(bytes).expect("build byte engine");
    let actual: Value = jetro
        .collect(query)
        .unwrap_or_else(|err| panic!("tape engine failed for {query}: {err}"));
    assert_eq!(actual, expected, "{query}");
}

#[test]
fn tape_matches_vm_for_borrowed_object_helpers() {
    let doc = json!({
        "profile": {
            "id": 7,
            "name": "Ada",
            "active": true,
            "nested": {"score": 42}
        }
    });
    for query in [
        "$.profile.keys()",
        "$.profile.values()",
        "$.profile.entries()",
        "$.profile.to_pairs()",
        "$.profile.entries().first()",
        "$.profile.to_pairs().last()",
        "$.profile.entries().last()",
        "$.profile.entries().map(e => e[0])",
        "$.profile.to_pairs().map(e => e[1]).last()",
        "$.profile.entries().map(e => e[1]).last()",
        "$.profile.entries().from_pairs()",
        "$.profile.pick(\"id\", \"name\")",
        "$.profile.omit(\"active\")",
        "$.profile.has_key(\"nested\")",
    ] {
        assert_tape_vm_eq(query, &doc);
    }
}

#[test]
fn tape_matches_vm_for_bounded_projection_and_reducers() {
    let doc = json!({
        "books": [
            {"id": 1, "title": "Dune", "price": 12.5, "tags": ["sf", "classic"]},
            {"id": 2, "title": "Foundation", "price": 9.0, "tags": ["sf"]},
            {"id": 3, "title": "Hyperion", "price": 14.0, "tags": ["sf", "hugo"]},
            {"id": 4, "title": "Snow Crash", "price": 11.0, "tags": ["sf", "cyberpunk"]}
        ]
    });
    for query in [
        "$.books.map({id, title}).first()",
        "$.books.map({id, title}).last()",
        "$.books.map({id, title}).take(2)",
        "$.books.filter(price > 10).map({id, title}).last()",
        "$.books.filter(tags.includes(\"hugo\")).map(price).sum()",
        "$.books.map(price).avg()",
        "$.books.map(price).min()",
        "$.books.map(price).max()",
        "$.books.count(price > 10)",
    ] {
        assert_tape_vm_eq(query, &doc);
    }
}

#[test]
fn tape_matches_vm_for_scalar_methods_inside_nested_projection() {
    let doc = json!({
        "users": [
            {
                "name": "ada",
                "email": "ada@example.test",
                "score": 10,
                "profile": {"role": "admin", "tier": 2}
            },
            {
                "name": "bob",
                "email": "bob@example.test",
                "score": 20,
                "profile": {"role": "user", "tier": 1}
            }
        ]
    });
    for query in [
        "$.users.map({name: name.upper(), domain: email.split(\"@\").last()})",
        "$.users.map({label: f\"{name}:{score}\", name_len: name.len()}).last()",
        "$.users.filter(name.starts_with(\"a\")).map(email.contains(\"@\"))",
        "$.users.map(profile.entries().first())",
        "$.users.map(profile.to_pairs().first())",
        "$.users.map(profile.entries().map(e => e[0]).last())",
        "$.users.map(profile.to_pairs().map(e => e[0]).last())",
    ] {
        assert_tape_vm_eq(query, &doc);
    }
}

#[test]
fn tape_matches_vm_for_scalar_methods_over_array_rows() {
    let doc = json!({
        "items": [
            {"name": " Ada ", "email": "ada@example.test", "score": -10.4, "code": "41"},
            {"name": "Bob", "email": "bob@example.org", "score": 20.5, "code": "x"},
            {"name": "Cy", "email": "cy@example.test", "score": -3.2, "code": "1"}
        ]
    });
    for query in [
        "$.items.map(name.upper()).last()",
        "$.items.map(name.trim().lower()).take(2)",
        "$.items.filter(name.starts_with(\" A\")).map(name.trim()).first()",
        "$.items.filter(email.ends_with(\".test\")).map(email.index_of(\"@\")).last()",
        "$.items.map(score.abs()).sum()",
        "$.items.map(score.round()).max()",
        "$.items.filter(code.is_numeric()).map(code.parse_int()).sum()",
        "$.items.map(name.byte_len()).sum()",
    ] {
        assert_tape_vm_eq(query, &doc);
    }
}

#[test]
fn tape_matches_vm_across_view_projection_boundaries() {
    let doc = json!({
        "users": [
            {
                "id": 1,
                "name": "ada",
                "profile": {
                    "role": "admin",
                    "contact": {"email": "ada@example.test"},
                    "flags": {"staff": true, "beta": null}
                }
            },
            {
                "id": 2,
                "name": "bob",
                "profile": {
                    "role": "user",
                    "contact": {"email": "bob@example.test"},
                    "flags": {"staff": false}
                }
            }
        ]
    });
    for query in [
        "$.users.map(profile.get_path(\"contact.email\").upper())",
        "$.users.map(profile.pick(\"role\", \"contact\").keys().last())",
        "$.users.map(profile.omit(\"flags\").values().first())",
        "$.users.map(profile.has_path(\"flags.staff\"))",
        "$.users.map(profile.missing(\"flags.beta\"))",
        "$.users.filter(profile.has_key(\"flags\")).map(profile.entries().last()[0])",
        "$.users.filter(profile.has_key(\"flags\")).map(profile.to_pairs().last()[0])",
        "$.users.map({id, email: profile.get_path(\"contact.email\"), keys: profile.keys().take(2)})",
    ] {
        assert_tape_vm_eq(query, &doc);
    }
}

#[test]
fn tape_matches_vm_for_object_helpers_over_array_rows() {
    let doc = json!({
        "teams": [
            {
                "name": "core",
                "members": [
                    {
                        "email": "ada@example.test",
                        "role": "lead",
                        "internal": true,
                        "addr": {"city": "Berlin", "zip": "10115"}
                    },
                    {
                        "email": "bob@example.test",
                        "role": "dev",
                        "internal": false,
                        "addr": {"city": "Paris"}
                    },
                    {
                        "email": "cy@example.test",
                        "role": "ops",
                        "internal": true,
                        "addr": {}
                    }
                ]
            }
        ]
    });
    for query in [
        "$.teams[0].members.pick(email, role).last()",
        "$.teams[0].members.omit(internal, addr).first()",
        "$.teams[0].members.filter(addr.has_path(\"city\")).map(addr.pick(\"city\", \"zip\")).first()",
        "$.teams[0].members.map({email, has_city: addr.has_key(\"city\"), city: addr.get_path(\"city\")}).last()",
        "$.teams[0].members.filter(addr.missing(\"zip\")).map(email).take(2)",
    ] {
        assert_tape_vm_eq(query, &doc);
    }
}

#[test]
fn tape_matches_vm_for_scalar_terminal_sinks() {
    let doc = json!({
        "books": [
            {"id": 1, "title": "Dune", "price": 12, "tags": ["sf", "classic"]},
            {"id": 2, "title": "Foundation", "price": 9, "tags": ["sf"]},
            {"id": 3, "title": "Hyperion", "price": 14, "tags": ["sf", "hugo"]},
            {"id": 4, "title": "Snow Crash", "price": 11, "tags": ["sf", "cyberpunk"]}
        ],
        "needle": "Hyperion"
    });
    for query in [
        "$.books.any(price > 13)",
        "$.books.all(tags.has(\"sf\"))",
        "$.books.find_index(title == $.needle)",
        "$.books.indices_where(price > 10)",
        "$.books.map(title).includes($.needle)",
        "$.books.map(title).index($.needle)",
        "$.books.map(tags.last()).indices_of(\"hugo\")",
        "$.books.filter(price > 10).map(title).includes(\"Dune\")",
        "$.books.filter(tags.includes(\"sf\")).map(price).any(@ > 13)",
    ] {
        assert_tape_vm_eq(query, &doc);
    }
}

#[test]
fn tape_matches_vm_for_keyed_reducers_and_distinct_stages() {
    let doc = json!({
        "orders": [
            {"id": "a1", "region": "eu", "status": "open", "total": 12, "customer": {"tier": "gold"}},
            {"id": "a2", "region": "us", "status": "open", "total": 9, "customer": {"tier": "silver"}},
            {"id": "a3", "region": "eu", "status": "closed", "total": 15, "customer": {"tier": "gold"}},
            {"id": "a4", "region": "apac", "status": "open", "total": 7, "customer": {"tier": "bronze"}},
            {"id": "a5", "region": "us", "status": "closed", "total": 20, "customer": {"tier": "silver"}}
        ]
    });
    for query in [
        "$.orders.count_by(region)",
        "$.orders.filter(total > 10).count_by(customer.tier)",
        "$.orders.index_by(id)",
        "$.orders.map({id, region, total}).index_by(id)",
        "$.orders.group_by(status)",
        "$.orders.group_by(region).entries().map({region: @[0], count: @[1].len()})",
        "$.orders.count_by(region).entries().map({region: @[0], count: @[1]}).last()",
        "$.orders.group_by(region).entries().map({region: @[0], total: @[1].sum(total), open: @[1].count(status == \"open\")})",
        "$.orders.group_by(customer.tier).entries().map({tier: @[0], ids: @[1].map(id).take(2)})",
        "$.orders.unique_by(region).map(id)",
        "$.orders.filter(status == \"open\").unique_by(customer.tier).map({id, tier: customer.tier})",
    ] {
        assert_tape_vm_eq(query, &doc);
    }
}

#[test]
fn tape_matches_vm_across_conservative_stage_boundaries() {
    let doc = json!({
        "orders": [
            {
                "id": "a1",
                "status": "open",
                "score": 30,
                "items": [
                    {"sku": "p1", "price": 12, "tags": ["hot", "fragile"]},
                    {"sku": "p2", "price": 4, "tags": ["cold"]}
                ]
            },
            {
                "id": "a2",
                "status": "open",
                "score": 20,
                "items": [
                    {"sku": "p3", "price": 9, "tags": ["hot"]},
                    {"sku": "p4", "price": 16, "tags": ["bulk", "hot"]}
                ]
            },
            {
                "id": "a3",
                "status": "closed",
                "score": 10,
                "items": [
                    {"sku": "p5", "price": 7, "tags": []}
                ]
            }
        ]
    });
    for query in [
        "$.orders.take_while(status == \"open\").map(id).last()",
        "$.orders.drop_while(score > 15).map({id, score}).first()",
        "$.orders.filter(status == \"open\").flat_map(items).filter(price > 10).map(sku).last()",
        "$.orders.flat_map(items).flat_map(tags).unique().last()",
        "$.orders.flat_map(items).map(tags.first()).unique().last()",
        "$.orders.sort_by(score).take(2).map({id, score}).last()",
        "$.orders.sort_by(score).drop(1).map(id).first()",
    ] {
        assert_tape_vm_eq(query, &doc);
    }
}

#[test]
fn tape_matches_vm_for_nested_reducers_inside_projection_bodies() {
    let doc = json!({
        "orders": [
            {
                "id": "a1",
                "status": "open",
                "items": [
                    {"sku": "p1", "price": 12, "tags": ["hot", "fragile"]},
                    {"sku": "p2", "price": 4, "tags": ["cold"]}
                ]
            },
            {
                "id": "a2",
                "status": "open",
                "items": [
                    {"sku": "p3", "price": 9, "tags": ["hot"]},
                    {"sku": "p4", "price": 16, "tags": ["bulk", "hot"]}
                ]
            },
            {
                "id": "a3",
                "status": "closed",
                "items": [
                    {"sku": "p5", "price": 7, "tags": []}
                ]
            }
        ]
    });
    for query in [
        "$.orders.map(items.map(price).sum()).last()",
        "$.orders.map({id, total: items.sum(price), max_price: items.max(price)}).last()",
        "$.orders.filter(items.any(price > 10)).map(items.filter(tags.has(\"hot\")).map(price).max()).last()",
        "$.orders.map(items.flat_map(tags).unique().last()).take(2)",
        "$.orders.map({id, hot: items.count(tags.has(\"hot\")), first_sku: items.map(sku).first()})",
    ] {
        assert_tape_vm_eq(query, &doc);
    }
}

#[test]
fn tape_matches_vm_for_object_helper_chains_inside_projection_bodies() {
    let doc = json!({
        "members": [
            {
                "email": "ada@example.test",
                "role": "lead",
                "addr": {"city": "Berlin", "zip": "10115"}
            },
            {
                "email": "bob@example.test",
                "role": "dev",
                "addr": {"city": "Paris"}
            },
            {
                "email": "cy@example.test",
                "role": "ops",
                "addr": {}
            }
        ]
    });
    for query in [
        "$.members.map(addr.values().map(@.to_string()).last())",
        "$.members.map(addr.entries().filter(@[0] != \"zip\").map(e => e[1]).last())",
        "$.members.map(addr.pick(\"city\", \"zip\").entries().map(e => e[0]).last())",
        "$.members.map({email, keys: addr.keys().take(2), has_zip: addr.has_key(\"zip\")}).last()",
        "$.members.filter(addr.has_key(\"city\")).map(addr.get_path(\"city\").upper()).first()",
    ] {
        assert_tape_vm_eq(query, &doc);
    }
}
