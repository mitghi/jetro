//! Parity coverage between the byte/tape engine and the VM fallback for
//! demand-sensitive pipeline shapes.

use super::common::vm_query;
use serde_json::{json, Value};

fn assert_tape_vm_eq(query: &str, doc: &Value) {
    let expected =
        vm_query(query, doc).unwrap_or_else(|err| panic!("VM failed for {query}: {err}"));
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
fn tape_matches_vm_for_object_lambda_helpers() {
    let doc = json!({
        "profiles": [
            {
                "id": 1,
                "settings": {
                    "theme": "dark",
                    "score": 10,
                    "_debug": true,
                    "feature_a": "on"
                }
            },
            {
                "id": 2,
                "settings": {
                    "theme": "light",
                    "score": 20,
                    "_debug": null,
                    "feature_b": "off"
                }
            },
            {
                "id": 3,
                "settings": {
                    "theme": "dark",
                    "score": 30,
                    "feature_c": "on"
                }
            }
        ]
    });
    for query in [
        "$.profiles.map(settings.transform_keys(k => k.upper())).last()",
        "$.profiles.map(settings.transform_values(v => v.to_string())).first()",
        "$.profiles.map(settings.filter_keys(k => not k.starts_with(\"_\"))).last()",
        "$.profiles.map(settings.filter_values(v => v != null)).take(2)",
        "$.profiles.filter(settings.filter_keys(k => k.starts_with(\"feature\")).len() > 0).map(id)",
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
fn tape_matches_vm_for_positional_demand_boundaries() {
    let doc = json!({
        "items": [
            {"id": 1, "name": "ada", "active": true, "score": 30},
            {"id": 2, "name": "bob", "active": false, "score": 20},
            {"id": 3, "name": "cy", "active": true, "score": 10},
            {"id": 4, "name": "dee", "active": true, "score": 40}
        ]
    });
    for query in [
        "$.items.map({id, label: name.upper()}).nth(2)",
        "$.items.filter(active).map({id, score}).nth(1)",
        "$.items.reverse().map(id).first()",
        "$.items.reverse().filter(active).map(name).last()",
        "$.items.skip(1).take(2).map(name.upper()).last()",
        "$.items.sort_by(score).skip(1).take(2).map({id, score}).first()",
    ] {
        assert_tape_vm_eq(query, &doc);
    }
}

#[test]
fn tape_matches_vm_for_nullary_and_literal_view_filters() {
    let doc = json!({
        "rows": [
            null,
            {"id": 1, "name": "ada", "tag": "keep"},
            {"id": 2, "name": "bob", "tag": "drop"},
            null,
            {"id": 3, "name": "cy", "tag": "keep"},
            {"id": 4, "name": "dee", "tag": "drop"}
        ],
        "tags": ["keep", "drop", null, "keep", "skip", null]
    });
    for query in [
        "$.rows.compact().map(id).last()",
        "$.rows.compact().filter(tag == \"keep\").map(name.upper()).take(2)",
        "$.rows.compact().remove({\"id\": 2, \"name\": \"bob\", \"tag\": \"drop\"}).map(id)",
        "$.tags.compact().remove(\"drop\").last()",
        "$.tags.remove(null).take(3)",
        "$.tags.remove(\"skip\").compact().count()",
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
fn tape_matches_vm_for_match_inside_demanded_chains() {
    let doc = json!({
        "events": [
            {"kind": "view", "name": "landing", "score": 5, "tags": ["web"]},
            {"kind": "click", "name": "cta", "score": 12, "x": 7, "tags": ["web", "conversion"]},
            {"kind": "click", "name": "nav", "score": 8, "x": 2, "tags": ["web", "nav"]},
            {"kind": "error", "name": "timeout", "score": 20, "tags": ["ops", "critical"]}
        ]
    });
    for query in [
        r#"$.events.map(match @ with {
            {kind: "click", x: x} -> {kind: "click", x: x},
            {kind: k} -> {kind: k},
            _ -> null
        }).last()"#,
        r#"$.events.filter(match @ with {
            {kind: "click", x: x} when x > 5 -> true,
            _ -> false
        }).map(name).first()"#,
        r#"$.events.map(match @ with {
            {tags: [first, ...rest]} -> rest.last(),
            _ -> null
        }).take(2)"#,
        r#"$.events.sort_by(score).take(2).map(match @ with {
            {kind: k, score: s} -> f"{k}:{s}",
            _ -> "?"
        }).last()"#,
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
fn tape_matches_vm_for_extreme_and_distinct_terminal_sinks() {
    let doc = json!({
        "books": [
            {"id": 1, "title": "Dune", "price": 12, "score": 91, "tag": "sf", "author": {"name": "Herbert"}},
            {"id": 2, "title": "Foundation", "price": 9, "score": 88, "tag": "sf", "author": {"name": "Asimov"}},
            {"id": 3, "title": "Hyperion", "price": 14, "score": 95, "tag": "hugo", "author": {"name": "Simmons"}},
            {"id": 4, "title": "Snow Crash", "price": 11, "score": 90, "tag": "cyber", "author": {"name": "Stephenson"}},
            {"id": 5, "title": "Neuromancer", "price": 10, "score": 93, "tag": "cyber", "author": {"name": "Gibson"}}
        ]
    });
    for query in [
        "$.books.map({title, value: score + price})",
        "$.books.map({title, value: score + price}).max_by(value)",
        "$.books.max_by(score).title",
        "$.books.min_by(price).author.name",
        "$.books.filter(tag == \"cyber\").max_by(score).title",
        "$.books.map({title, value: score + price}).max_by(value).title",
        "$.books.sort_by(-score).first().title",
        "$.books.sort_by(-score).last().title",
        "$.books.sort_by(-score).take(2).map({title, score}).last().title",
        "$.books.map(tag).approx_count_distinct()",
        "$.books.filter(score > 90).map(author.name).approx_count_distinct()",
    ] {
        assert_tape_vm_eq(query, &doc);
    }
}

#[test]
fn tape_matches_vm_for_window_and_series_stages() {
    let doc = json!({
        "samples": [
            {"id": "a", "value": 10},
            {"id": "b", "value": 13},
            {"id": "c", "value": 18},
            {"id": "d", "value": 12},
            {"id": "e", "value": 20}
        ]
    });
    for query in [
        "$.samples.map(value).window(3).last()",
        "$.samples.map(value).chunk(2).take(2)",
        "$.samples.map(value).lag(1).last()",
        "$.samples.map(value).lead(1).first()",
        "$.samples.map(value).rolling_sum(2).last()",
        "$.samples.map(value).rolling_avg(3).last()",
        "$.samples.map(value).diff_window().take(3)",
        "$.samples.map(value).pct_change().last()",
        "$.samples.map(value).cummax().last()",
        "$.samples.map(value).cummin().last()",
        "$.samples.map(value).zscore().take(2)",
    ] {
        assert_tape_vm_eq(query, &doc);
    }
}

#[test]
fn tape_matches_vm_for_materialized_suffix_after_view_prefix() {
    let doc = json!({
        "samples": [
            {"id": "a", "value": 10},
            {"id": "b", "value": 13},
            {"id": "c", "value": 18},
            {"id": "d", "value": 12},
            {"id": "e", "value": 20}
        ]
    });
    for query in [
        "$.samples.map(value).partition(@ > 12).last()",
        "$.samples.map(value).accumulate((a, b) => a + b).last()",
        "$.samples.map(value).append(99).last()",
        "$.samples.map(value).prepend(5).first()",
        "$.samples.map(value).diff([13, 20]).last()",
        "$.samples.map(value).intersect([10, 18, 99]).last()",
        "$.samples.map(value).union([10, 99]).last()",
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
        "$.orders.unique_by(status).map({id, hot_sku: items.filter(tags.has(\"hot\")).map(sku).last()}).last()",
        "$.orders.take_while(status == \"open\").map({id, item: items.last().pick(\"sku\", \"price\")}).last()",
        "$.orders.drop_while(score > 15).flat_map(items).map({sku, tag: tags.first()}).first()",
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

#[test]
fn tape_matches_vm_for_sparse_object_helpers_and_find_sinks() {
    let doc = json!({
        "rows": [
            {"id": 1, "score": 30, "meta": {"isbn": "a", "price": 12, "author": {"name": "Ada"}}},
            {"id": 2, "score": 10, "meta": null},
            {"id": 3, "score": 20, "meta": {"price": 9, "author": {"name": "Bob"}}},
            {"id": 4, "score": 40},
            {"id": 5, "score": 50, "meta": {"isbn": "c", "price": 16, "author": {"name": "Cy"}}}
        ]
    });
    for query in [
        "$.rows.find(meta.has_key(\"isbn\")).meta.pick(\"isbn\", \"price\")",
        "$.rows.find_all(meta.has_key(\"isbn\")).map(meta.get_path(\"isbn\")).last()",
        "$.rows.filter(meta.has_path(\"author.name\")).map(meta.pick(\"author\").get_path(\"author.name\").upper()).first()",
        "$.rows.filter(meta.missing(\"isbn\")).map({id, keys: meta.keys()}).take(2)",
        "$.rows.sort_by(score).drop_while(meta.missing(\"isbn\")).map({id, isbn: meta.get_path(\"isbn\")}).first()",
        "$.rows.filter(meta.has_key(\"isbn\")).map({id, value: meta.values().last(), entry: meta.entries().first()}).last()",
    ] {
        assert_tape_vm_eq(query, &doc);
    }
}

#[test]
fn tape_matches_vm_for_late_projection_after_mixed_boundaries() {
    let doc = json!({
        "orders": [
            {
                "id": "a1",
                "score": 30,
                "status": "open",
                "customer": {"name": "Ada", "tier": "gold"},
                "items": [
                    {"sku": "p1", "price": 12, "qty": 2, "tags": ["hot", "fragile"]},
                    {"sku": "p2", "price": 4, "qty": 1, "tags": ["cold"]}
                ]
            },
            {
                "id": "a2",
                "score": 10,
                "status": "closed",
                "customer": {"name": "Bob", "tier": "silver"},
                "items": [
                    {"sku": "p3", "price": 9, "qty": 3, "tags": ["hot"]},
                    {"sku": "p4", "price": 16, "qty": 1, "tags": ["bulk", "hot"]}
                ]
            },
            {
                "id": "a3",
                "score": 20,
                "status": "open",
                "customer": {"name": "Cy", "tier": "gold"},
                "items": [
                    {"sku": "p5", "price": 7, "qty": 4, "tags": []}
                ]
            },
            {
                "id": "a4",
                "score": 40,
                "status": "open",
                "customer": {"name": "Dee", "tier": "bronze"},
                "items": [
                    {"sku": "p6", "price": 20, "qty": 1, "tags": ["hot"]},
                    {"sku": "p7", "price": 3, "qty": 5, "tags": ["clearance"]}
                ]
            }
        ]
    });
    for query in [
        "$.orders.map({id, gross: items.map(price * qty).sum(), first_hot: items.find(tags.has(\"hot\")).sku}).last()",
        "$.orders.filter(status == \"open\").map({id, name: customer.name.upper(), total: items.sum(price)}).nth(1)",
        "$.orders.sort_by(score).take(2).map({id, top_sku: items.sort_by(-price).first().sku, tag_count: items.flat_map(tags).count()}).last()",
        "$.orders.drop_while(score > 25).map({id, keys: customer.keys(), tier: customer.get_path(\"tier\")}).first()",
        "$.orders.take_while(status == \"open\").map({id, item: items.last().pick(\"sku\", \"price\")}).last()",
        "$.orders.filter(items.any(price > 15)).map({id, match: items.filter(price > 10).map(sku).first()}).last()",
    ] {
        assert_tape_vm_eq(query, &doc);
    }
}

#[test]
fn tape_matches_vm_for_terminal_sinks_after_materialized_boundaries() {
    let doc = json!({
        "orders": [
            {
                "id": "a1",
                "status": "open",
                "score": 30,
                "customer": {"region": "eu", "tier": "gold"},
                "items": [
                    {"sku": "p1", "price": 12, "qty": 2, "tags": ["hot", "fragile"]},
                    {"sku": "p2", "price": 4, "qty": 1, "tags": ["cold"]}
                ]
            },
            {
                "id": "a2",
                "status": "closed",
                "score": 10,
                "customer": {"region": "us", "tier": "silver"},
                "items": [
                    {"sku": "p3", "price": 9, "qty": 3, "tags": ["hot"]},
                    {"sku": "p4", "price": 16, "qty": 1, "tags": ["bulk", "hot"]}
                ]
            },
            {
                "id": "a3",
                "status": "open",
                "score": 20,
                "customer": {"region": "eu", "tier": "gold"},
                "items": [
                    {"sku": "p5", "price": 7, "qty": 4, "tags": []}
                ]
            },
            {
                "id": "a4",
                "status": "open",
                "score": 40,
                "customer": {"region": "apac", "tier": "bronze"},
                "items": [
                    {"sku": "p6", "price": 20, "qty": 1, "tags": ["hot"]},
                    {"sku": "p7", "price": 3, "qty": 5, "tags": ["clearance"]}
                ]
            }
        ],
        "wanted_sku": "p6",
        "wanted_region": "eu"
    });
    for query in [
        "$.orders.flat_map(items).map(sku).includes($.wanted_sku)",
        "$.orders.flat_map(items).map(sku).index($.wanted_sku)",
        "$.orders.flat_map(items).map(tags.first()).indices_of(\"hot\")",
        "$.orders.sort_by(score).drop(1).map(customer.region).includes($.wanted_region)",
        "$.orders.sort_by(score).drop(1).map(customer.region).indices_of($.wanted_region)",
        "$.orders.take_while(status == \"open\").map(items.map(price).sum()).any(@ > 20)",
        "$.orders.drop_while(score > 25).map({id, total: items.sum(price)}).find_index(total > 20)",
        "$.orders.flat_map(items).indices_where(tags.includes(\"hot\"))",
        "$.orders.group_by(customer.region).entries().map({region: @[0], count: @[1].len()}).any(count > 1)",
        "$.orders.unique_by(customer.tier).map(customer.region).all(@ != \"\")",
    ] {
        assert_tape_vm_eq(query, &doc);
    }
}

#[test]
fn tape_matches_vm_for_dynamic_membership_targets_in_view_chains() {
    let doc = json!({
        "rows": [
            {"id": 1, "name": "ada", "tags": ["admin", "ops"], "meta": {"primary": "admin"}},
            {"id": 2, "name": "bob", "tags": ["user"], "meta": {"primary": "user"}},
            {"id": 3, "name": "cy", "tags": ["ops", "review"], "meta": {"primary": "review"}},
            {"id": 4, "name": "dee", "tags": [], "meta": {"primary": "guest"}}
        ],
        "needles": {
            "role": "ops",
            "empty": [],
            "compound": {"id": 3, "tag": "review"}
        }
    });
    for query in [
        "$.rows.map(tags).includes([\"user\"])",
        "$.rows.map({id, tag: tags.last()}).map(tag).includes($.needles.role)",
        "$.rows.map(tags.first()).includes($.needles.role)",
        "$.rows.map(tags.first()).index($.needles.role)",
        "$.rows.map(meta.primary).indices_of($.needles.role)",
        "$.rows.filter(tags.includes($.needles.role)).map(name).includes(\"ada\")",
        "$.rows.filter(tags.includes($.needles.role)).map({id, tag: tags.last()}).index($.needles.compound)",
        "$.rows.map(tags.len()).indices_of(0)",
    ] {
        assert_tape_vm_eq(query, &doc);
    }
}
