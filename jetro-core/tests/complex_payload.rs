

use jetro_core::Jetro;
use serde_json::{json, Value};

const N_ORDERS: usize = 2_000;
const ITEMS_PER_ORDER: usize = 6;

fn j(document: Value) -> Jetro {
    Jetro::from_bytes(serde_json::to_vec(&document).unwrap()).unwrap()
}

fn synth_doc() -> Value {
    let regions = [
        "us-east",
        "us-west",
        "eu-central",
        "ap-southeast",
        "sa-south",
    ];
    let statuses = ["pending", "shipped", "delivered", "cancelled", "refunded"];
    let priorities = ["low", "normal", "high", "urgent"];
    let cities = [
        "Tokyo",
        "Berlin",
        "São Paulo",
        "Nairobi",
        "Austin",
        "Toronto",
    ];
    let country = ["JP", "DE", "BR", "KE", "US", "CA"];

    let mut orders = Vec::with_capacity(N_ORDERS);
    for i in 0..N_ORDERS {
        let mut items = Vec::with_capacity(ITEMS_PER_ORDER);
        let mut total: f64 = 0.0;
        for j in 0..ITEMS_PER_ORDER {
            let price = ((i * 7 + j * 13) % 500) as f64 + 9.99;
            let qty = ((i + j) % 5 + 1) as i64;
            total += price * qty as f64;
            items.push(json!({
                "sku":   format!("SKU-{:05}", (i * ITEMS_PER_ORDER + j) % 9973),
                "name":  format!("item-{}-{}", i, j),
                "price": price,
                "qty":   qty,
            }));
        }
        orders.push(json!({
            "id":       100_000 + i,
            "status":   statuses[i % statuses.len()],
            "priority": priorities[(i / 3) % priorities.len()],
            "region":   regions[i % regions.len()],
            "total":    (total * 100.0).round() / 100.0,
            "customer": {
                "id":   10_000 + (i % 500),
                "name": format!("Customer {}", i % 500),
                "address": {
                    "city":         cities[i % cities.len()],
                    "country_code": country[i % country.len()],
                }
            },
            "items": items,
        }));
    }
    json!({ "orders": orders, "meta": { "kind": "complex_payload" } })
}

fn as_array(v: &Value) -> &Vec<Value> {
    v.as_array().expect("expected array")
}


#[test]
fn q1_project_nested_field() {
    let j = j(synth_doc());
    let out = j.collect("$.orders.map(customer.address.city)").unwrap();
    let arr = as_array(&out);
    assert_eq!(arr.len(), N_ORDERS);
    
    let cities = [
        "Tokyo",
        "Berlin",
        "São Paulo",
        "Nairobi",
        "Austin",
        "Toronto",
    ];
    for v in arr {
        let s = v.as_str().unwrap();
        assert!(cities.contains(&s), "unexpected city: {}", s);
    }
}

#[test]
fn q2_project_then_unique() {
    let j = j(synth_doc());
    let out = j
        .collect("$.orders.map(customer.address.country_code).unique()")
        .unwrap();
    let arr = as_array(&out);
    assert_eq!(arr.len(), 6); 
}


#[test]
fn q3_filter_then_map_id() {
    let j = j(synth_doc());
    let out = j.collect("$.orders.filter(total > 500).map(id)").unwrap();
    
    let arr = as_array(&out);
    assert!(!arr.is_empty());
    for v in arr {
        assert!(v.is_i64(), "id not int: {:?}", v);
    }
}

#[test]
fn q4_multi_cond_filter_count_matches_naive() {
    let doc = synth_doc();
    let naive: usize = doc["orders"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|o| o["status"] == "shipped" && o["priority"] == "high")
        .count();
    let j = j(doc);
    let out = j
        .collect(r#"$.orders.filter(status == "shipped" and priority == "high").count()"#)
        .unwrap();
    assert_eq!(out.as_i64().unwrap() as usize, naive);
}


#[test]
fn q5_deep_find_broad_tree_eq_scan() {
    let doc = synth_doc();
    let bytes = serde_json::to_vec(&doc).unwrap();
    let j_tree = j(doc);
    let j_scan = Jetro::from_bytes(bytes).unwrap();
    let q = r#"$..find(@.status == "shipped")"#;
    let t = j_tree.collect(q).unwrap();
    let s = j_scan.collect(q).unwrap();
    
    assert_eq!(as_array(&t).len(), as_array(&s).len());
    assert!(!as_array(&t).is_empty());
}

#[test]
fn q6_deep_find_narrow_single_hit() {
    let doc = synth_doc();
    let bytes = serde_json::to_vec(&doc).unwrap();
    let j_tree = j(doc);
    let j_scan = Jetro::from_bytes(bytes).unwrap();
    let q = r#"$..find(@.sku == "SKU-00042")"#;
    let t = as_array(&j_tree.collect(q).unwrap()).len();
    let s = as_array(&j_scan.collect(q).unwrap()).len();
    assert_eq!(t, s);
}

#[test]
fn q7_deep_find_multi_predicate_and() {
    let doc = synth_doc();
    let bytes = serde_json::to_vec(&doc).unwrap();
    let j_tree = j(doc);
    let j_scan = Jetro::from_bytes(bytes).unwrap();
    let q = r#"$..find(@.status == "shipped", @.priority == "urgent")"#;
    let t = as_array(&j_tree.collect(q).unwrap()).len();
    let s = as_array(&j_scan.collect(q).unwrap()).len();
    assert_eq!(t, s);
    
    let q_broad = r#"$..find(@.status == "shipped")"#;
    let broad = as_array(&j_tree.collect(q_broad).unwrap()).len();
    assert!(t <= broad);
}


#[test]
fn q8_deep_key_sum_tree_eq_scan() {
    let doc = synth_doc();
    let bytes = serde_json::to_vec(&doc).unwrap();
    let j_tree = j(doc);
    let j_scan = Jetro::from_bytes(bytes).unwrap();
    let q = "$..total.sum()";
    let t = j_tree.collect(q).unwrap();
    let s = j_scan.collect(q).unwrap();
    
    let tf = t.as_f64().expect("tree sum is number");
    let sf = s.as_f64().expect("scan sum is number");
    assert!((tf - sf).abs() < 1e-6, "tree {} vs scan {}", tf, sf);
}

#[test]
fn q9_deep_key_extract_sku_count() {
    let doc = synth_doc();
    let bytes = serde_json::to_vec(&doc).unwrap();
    let j_tree = j(doc);
    let j_scan = Jetro::from_bytes(bytes).unwrap();
    let q = "$..sku";
    let t_len = as_array(&j_tree.collect(q).unwrap()).len();
    let s_len = as_array(&j_scan.collect(q).unwrap()).len();
    assert_eq!(t_len, s_len);
    assert_eq!(t_len, N_ORDERS * ITEMS_PER_ORDER);
}


#[test]
fn q10_group_by_status_partition() {
    let j = j(synth_doc());
    let out = j.collect("$.orders.group_by(status)").unwrap();
    
    
    let obj = out.as_object().expect("object keyed by group value");
    let mut total = 0usize;
    for (_k, bucket) in obj {
        total += bucket.as_array().unwrap().len();
    }
    assert_eq!(total, N_ORDERS);
    assert_eq!(obj.len(), 5);
}

#[test]
fn q10_group_by_status_collect_matches() {
    let j = j(synth_doc());
    let v = j.collect("$.orders.group_by(status)").unwrap();
    let obj = v.as_object().expect("group_by did not return an object");
    assert_eq!(obj.len(), 5);
    let total: usize = obj
        .values()
        .map(|b| b.as_array().expect("bucket is not an array").len())
        .sum();
    assert_eq!(total, N_ORDERS);
}

#[test]
fn q11_count_by_region() {
    let j = j(synth_doc());
    let out = j.collect("$.orders.count_by(region)").unwrap();
    let obj = out.as_object().expect("object");
    let total: i64 = obj.values().map(|v| v.as_i64().unwrap()).sum();
    assert_eq!(total as usize, N_ORDERS);
    assert_eq!(obj.len(), 5); 
}

#[test]
fn q12_sum_of_totals_matches_naive() {
    let doc = synth_doc();
    let naive: f64 = doc["orders"]
        .as_array()
        .unwrap()
        .iter()
        .map(|o| o["total"].as_f64().unwrap())
        .sum();
    let j = j(doc);
    let out = j.collect("$.orders.map(total).sum()").unwrap();
    let got = out.as_f64().unwrap();
    assert!((got - naive).abs() < 1e-3, "got {} vs naive {}", got, naive);
}

#[test]
fn q15_max_matches_naive() {
    let doc = synth_doc();
    let naive = doc["orders"]
        .as_array()
        .unwrap()
        .iter()
        .map(|o| o["total"].as_f64().unwrap())
        .fold(f64::MIN, f64::max);
    let j = j(doc);
    let out = j.collect("$.orders.map(total).max()").unwrap();
    assert!((out.as_f64().unwrap() - naive).abs() < 1e-6);
}


#[test]
fn q13_list_comp_equivalent_to_filter_map() {
    let doc = synth_doc();
    let j = j(doc);
    let a = j
        .collect("[o.id for o in $.orders if o.total > 1000]")
        .unwrap();
    let b = j.collect("$.orders.filter(total > 1000).map(id)").unwrap();
    assert_eq!(a, b);
}

#[test]
fn q14_pick_projects_and_renames() {
    
    
    let j = j(synth_doc());
    let out = j
        .collect("$.orders.map(customer).pick(uid: id, who: name)")
        .unwrap();
    let arr = as_array(&out);
    assert_eq!(arr.len(), N_ORDERS);
    let first = arr[0].as_object().unwrap();
    for key in ["uid", "who"] {
        assert!(first.contains_key(key), "missing key {}", key);
    }
    assert!(!first.contains_key("id"));
    assert!(!first.contains_key("name"));
}


#[test]
fn q16_set_deep_address_replaces_leaf_obj() {
    let j = j(synth_doc());
    let out = j.collect(
        r#"$.orders[0].customer.address.set({"city": "Remote", "zip": "00000", "country_code": "XX", "street": "N/A"})"#
    ).unwrap();
    let city = &out["orders"][0]["customer"]["address"]["city"];
    assert_eq!(city.as_str(), Some("Remote"));
    
    let city2 = &out["orders"][1]["customer"]["address"]["city"];
    assert_ne!(city2.as_str(), Some("Remote"));
}

#[test]
fn q17_modify_nested_numeric_field() {
    let doc = synth_doc();
    let before = doc["orders"][0]["total"].as_f64().unwrap();
    let j = j(doc);
    let out = j.collect("$.orders[0].total.modify(@ * 2)").unwrap();
    let after = out["orders"][0]["total"].as_f64().unwrap();
    assert!((after - before * 2.0).abs() < 1e-6);
}

#[test]
fn q18_set_deep_items_array_resets() {
    let j = j(synth_doc());
    let out = j.collect("$.orders[0].items[0].price.set(0)").unwrap();
    assert_eq!(out["orders"][0]["items"][0]["price"].as_i64(), Some(0));
    
    assert_ne!(out["orders"][0]["items"][1]["price"].as_i64(), Some(0));
}


#[test]
fn route_c_scan_agrees_with_tree_walker_on_chained_find() {
    let doc = synth_doc();
    let bytes = serde_json::to_vec(&doc).unwrap();
    let j_tree = j(doc);
    let j_scan = Jetro::from_bytes(bytes).unwrap();
    
    let q = "$..total.sum()";
    let t = j_tree.collect(q).unwrap().as_f64().unwrap();
    let s = j_scan.collect(q).unwrap().as_f64().unwrap();
    assert!((t - s).abs() < 1e-6);
}

#[test]
fn find_count_fusion_yields_same_integer_as_filter_count() {
    let j = j(synth_doc());
    let a = j
        .collect(r#"$.orders.find(status == "shipped").count()"#)
        .unwrap();
    let b = j
        .collect(r#"$.orders.filter(status == "shipped").count()"#)
        .unwrap();
    assert_eq!(a, b);
}

#[test]
fn filter_map_min_max_match_unfused_pipeline() {
    let j = j(synth_doc());
    let fused_min = j
        .collect(r#"$.orders.filter(status == "shipped").map(total).min()"#)
        .unwrap();
    let fused_max = j
        .collect(r#"$.orders.filter(status == "shipped").map(total).max()"#)
        .unwrap();
    let arr: Vec<f64> = j
        .collect(r#"$.orders.filter(status == "shipped").map(total)"#)
        .unwrap()
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect();
    let naive_min = arr.iter().cloned().fold(f64::INFINITY, f64::min);
    let naive_max = arr.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    assert!((fused_min.as_f64().unwrap() - naive_min).abs() < 1e-9);
    assert!((fused_max.as_f64().unwrap() - naive_max).abs() < 1e-9);
}

#[test]
fn deep_find_numeric_range_tree_eq_scan() {
    let doc = synth_doc();
    let bytes = serde_json::to_vec(&doc).unwrap();
    let j_tree = j(doc.clone());
    let j_scan = Jetro::from_bytes(bytes).unwrap();
    
    
    for q in [
        "$..find(@.total > 500)",
        "$..find(@.qty < 3)",
        
        "$..find(500 < @.total)",
    ] {
        let t = as_array(&j_tree.collect(q).unwrap()).len();
        let s = as_array(&j_scan.collect(q).unwrap()).len();
        assert_eq!(t, s, "query {}: tree {} vs scan {}", q, t, s);
        assert!(t > 0, "query {} returned empty", q);
    }
    
    let orders = doc["orders"].as_array().unwrap();
    let naive_total_gte_500 = orders
        .iter()
        .filter(|o| o["total"].as_f64().unwrap() >= 500.0)
        .count();
    let scan_total_gte_500 = as_array(&j_scan.collect("$..find(@.total >= 500)").unwrap()).len();
    assert_eq!(scan_total_gte_500, naive_total_gte_500);

    let naive_qty_lte_2: usize = orders
        .iter()
        .map(|o| {
            o["items"]
                .as_array()
                .unwrap()
                .iter()
                .filter(|it| it["qty"].as_i64().unwrap() <= 2)
                .count()
        })
        .sum();
    let scan_qty_lte_2 = as_array(&j_scan.collect("$..find(@.qty <= 2)").unwrap()).len();
    assert_eq!(scan_qty_lte_2, naive_qty_lte_2);
}

#[test]
fn deep_find_then_count_and_aggregate_projection() {
    
    
    let doc = synth_doc();
    let bytes = serde_json::to_vec(&doc).unwrap();
    let j_tree = j(doc.clone());
    let j_scan = Jetro::from_bytes(bytes).unwrap();
    for q in [
        r#"$..find(@.status == "shipped").count()"#,
        r#"$..find(@.total > 500).count()"#,
        r#"$..find(@.status == "shipped").map(total).sum()"#,
        r#"$..find(@.status == "shipped").map(total).min()"#,
        r#"$..find(@.status == "shipped").map(total).max()"#,
        r#"$..find(@.status == "shipped").map(total).avg()"#,
    ] {
        let t = j_tree.collect(q).unwrap();
        let s = j_scan.collect(q).unwrap();
        
        let eps = 1e-6_f64;
        let tf = t.as_f64();
        let sf = s.as_f64();
        match (tf, sf) {
            (Some(a), Some(b)) => assert!(
                (a - b).abs() < eps.max(a.abs() * 1e-9),
                "query {}: tree {} vs scan {}",
                q,
                a,
                b
            ),
            _ => assert_eq!(t, s, "query {}", q),
        }
    }
}

#[test]
fn deep_find_then_map_field_direct_extract() {
    
    
    let doc = synth_doc();
    let bytes = serde_json::to_vec(&doc).unwrap();
    let j_tree = j(doc.clone());
    let j_scan = Jetro::from_bytes(bytes).unwrap();
    for q in [
        r#"$..find(@.status == "shipped").map(total)"#,
        r#"$..find(@.total > 500).map(id)"#,
        r#"$..find(@.status == "shipped", @.total > 500).map(region)"#,
    ] {
        let t: Vec<Value> = as_array(&j_tree.collect(q).unwrap()).clone();
        let s: Vec<Value> = as_array(&j_scan.collect(q).unwrap()).clone();
        assert_eq!(t.len(), s.len(), "len differs for {}", q);
        
        
        let mut ts: Vec<String> = t.iter().map(|v| v.to_string()).collect();
        let mut ss: Vec<String> = s.iter().map(|v| v.to_string()).collect();
        ts.sort();
        ss.sort();
        assert_eq!(ts, ss, "multiset differs for {}", q);
    }
}

#[test]
fn deep_find_mixed_eq_cmp_tree_eq_scan() {
    
    
    let doc = synth_doc();
    let bytes = serde_json::to_vec(&doc).unwrap();
    let j_tree = j(doc.clone());
    let j_scan = Jetro::from_bytes(bytes).unwrap();
    let q = r#"$..find(@.status == "shipped", @.total > 500)"#;
    let t = as_array(&j_tree.collect(q).unwrap()).len();
    let s = as_array(&j_scan.collect(q).unwrap()).len();
    assert_eq!(t, s, "tree {} vs scan {}", t, s);
    let orders = doc["orders"].as_array().unwrap();
    let naive = orders
        .iter()
        .filter(|o| o["status"].as_str() == Some("shipped") && o["total"].as_f64().unwrap() > 500.0)
        .count();
    assert_eq!(s, naive);
    assert!(s > 0);
}

#[test]
fn descendant_first_early_exit_matches_tree() {
    
    
    let doc = synth_doc();
    let bytes = serde_json::to_vec(&doc).unwrap();
    let j_tree = j(doc.clone());
    let j_scan = Jetro::from_bytes(bytes).unwrap();
    for q in [
        "$..sku.first()",
        "$..qty.first()",
        "$..items.first()..sku.first()",
    ] {
        assert_eq!(
            j_scan.collect(q).unwrap(),
            j_tree.collect(q).unwrap(),
            "query {}",
            q
        );
    }
    
    
    let first_sku = j_scan.collect("$..sku.first()").unwrap();
    assert!(first_sku.as_str().unwrap_or("").starts_with("SKU-"));
}

#[test]
fn unique_count_fusion_matches_dedup_then_count() {
    let j = j(synth_doc());
    let fused = j.collect("$.orders.map(status).unique().count()").unwrap();
    let plain = j.collect("$.orders.map(status).unique().len()").unwrap();
    let manual = {
        let arr = j.collect("$.orders.map(status).unique()").unwrap();
        serde_json::Value::from(arr.as_array().unwrap().len() as i64)
    };
    assert_eq!(fused, manual);
    assert_eq!(plain, manual);
}
