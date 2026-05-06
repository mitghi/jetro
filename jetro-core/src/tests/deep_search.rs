//! Tests for deep-search constructs: `$..find`, `$..shape`, `$..like`, and
//! the SIMD-accelerated descendant scan path (route C in the planner).

#[cfg(test)]
mod tests {
    use super::super::common::{books, vm_query};
    use serde_json::json;

    #[test]
    fn simd_scan_descendant_matches_tree_walker() {
        use crate::Jetro;
        let raw = br#"{"a":{"test":1},"b":[{"test":2},{"other":9},{"test":3}],"comment":"the \"test\": lie"}"#.to_vec();
        let j_bytes = Jetro::from_bytes(raw.clone()).unwrap();
        let j_tree = Jetro::new(serde_json::from_slice(&raw).unwrap());
        assert_eq!(
            j_bytes.collect("$..test").unwrap(),
            j_tree.collect("$..test").unwrap()
        );
    }

    #[test]
    fn simd_scan_chains_further_steps() {
        use crate::Jetro;
        let raw =
            br#"{"users":[{"id":1,"name":"a"},{"id":2,"name":"b"},{"id":3,"name":"c"}]}"#.to_vec();
        let j = Jetro::from_bytes(raw).unwrap();
        let r = j.collect("$..id.sum()").unwrap();
        assert_eq!(r, json!(6));
    }

    #[test]
    fn simd_scan_via_vm_path() {
        
        
        use crate::Jetro;
        let raw = br#"{"a":{"x":1},"b":[{"x":2},{"x":3}]}"#.to_vec();
        let j_b = Jetro::from_bytes(raw.clone()).unwrap();
        let j_t = Jetro::new(serde_json::from_slice(&raw).unwrap());
        
        assert_eq!(j_b.collect("$..x").unwrap(), j_t.collect("$..x").unwrap());
        assert_eq!(j_b.collect("$..x").unwrap(), j_t.collect("$..x").unwrap());
    }

    #[test]
    fn simd_scan_vm_path_aggregate() {
        
        
        use crate::Jetro;
        let raw = br#"{"rows":[{"p":10},{"p":20},{"p":30}]}"#.to_vec();
        let j = Jetro::from_bytes(raw).unwrap();
        assert_eq!(j.collect("$..p.sum()").unwrap(), json!(60));
    }

    #[test]
    fn simd_scan_literal_eq_int() {
        
        
        use crate::Jetro;
        let doc = json!({"xs":[{"n":10},{"n":42},{"n":10},{"n":42},{"n":7}]});
        let raw = serde_json::to_vec(&doc).unwrap();
        let j = Jetro::from_bytes(raw).unwrap();
        assert_eq!(j.collect("$..n.filter(@ == 42)").unwrap(), json!([42, 42]));
    }

    #[test]
    fn simd_scan_literal_eq_string() {
        use crate::Jetro;
        let doc = json!({
            "events":[
                {"type":"action"},{"type":"idle"},
                {"type":"action"},{"type":"noop"}
            ]
        });
        let raw = serde_json::to_vec(&doc).unwrap();
        let j = Jetro::from_bytes(raw).unwrap();
        assert_eq!(
            j.collect(r#"$..type.filter(@ == "action")"#).unwrap(),
            json!(["action", "action"])
        );
    }

    #[test]
    fn simd_scan_literal_eq_bool_null() {
        use crate::Jetro;
        let doc = json!({"xs":[{"v":true},{"v":false},{"v":true},{"v":null}]});
        let raw = serde_json::to_vec(&doc).unwrap();
        let j = Jetro::from_bytes(raw.clone()).unwrap();
        assert_eq!(
            j.collect("$..v.filter(@ == true)").unwrap(),
            json!([true, true])
        );

        let j2 = Jetro::from_bytes(raw).unwrap();
        assert_eq!(j2.collect("$..v.filter(@ == null)").unwrap(), json!([null]));
    }

    #[test]
    fn route_c_chained_descendants_match_tree_walker() {
        
        
        use crate::Jetro;
        let doc = json!({
            "outer":[
                {"inner":[{"leaf":1},{"leaf":2}]},
                {"inner":[{"leaf":3},{"leaf":4}]}
            ]
        });
        let raw = serde_json::to_vec(&doc).unwrap();
        let j_b = Jetro::from_bytes(raw).unwrap();
        let j_t = Jetro::new(doc);
        let q = "$..outer.first()..inner.first()..leaf";
        assert_eq!(j_b.collect(q).unwrap(), j_t.collect(q).unwrap());
        assert_eq!(j_b.collect(q).unwrap(), json!([1, 2]));
    }

    #[test]
    fn route_c_descendant_after_filter_eq() {
        
        use crate::Jetro;
        let doc = json!({
            "items":[
                {"kind":"a","children":[{"v":1},{"v":2}]},
                {"kind":"b","children":[{"v":3},{"v":4}]}
            ]
        });
        let raw = serde_json::to_vec(&doc).unwrap();
        let j_b = Jetro::from_bytes(raw).unwrap();
        let j_t = Jetro::new(doc);
        let q = r#"$..kind.filter(@ == "a")"#;
        assert_eq!(j_b.collect(q).unwrap(), j_t.collect(q).unwrap());
        assert_eq!(j_b.collect(q).unwrap(), json!(["a"]));
    }

    #[test]
    fn route_c_quantifier_scalar_result() {
        
        use crate::Jetro;
        let doc = json!({"xs":[{"id":7},{"id":8}]});
        let raw = serde_json::to_vec(&doc).unwrap();
        let j_b = Jetro::from_bytes(raw).unwrap();
        assert_eq!(j_b.collect("$..id.first()").unwrap(), json!(7));
    }

    #[test]
    fn deep_find_field_eq_scan_matches_tree_walker() {
        
        
        use crate::Jetro;
        let doc = json!({
            "a":[
                {"type":"action","v":1},
                {"type":"idle","v":2},
                {"nested":{"type":"action","v":3}}
            ],
            "b":{"type":"action","v":4}
        });
        let raw = serde_json::to_vec(&doc).unwrap();
        let j_b = Jetro::from_bytes(raw).unwrap();
        let j_t = Jetro::new(doc);
        let q = r#"$..find(@.type == "action")"#;
        let rb = j_b.collect(q).unwrap();
        let rt = j_t.collect(q).unwrap();
        assert_eq!(rb, rt);
        let arr = rb.as_array().unwrap();
        assert_eq!(arr.len(), 3);
    }

    #[test]
    fn deep_find_field_eq_int_literal() {
        use crate::Jetro;
        let doc = json!({"xs":[{"id":1,"t":"a"},{"id":2,"t":"b"},{"deep":{"id":1,"t":"c"}}]});
        let raw = serde_json::to_vec(&doc).unwrap();
        let j_b = Jetro::from_bytes(raw).unwrap();
        let j_t = Jetro::new(doc);
        let q = "$..find(@.id == 1)";
        assert_eq!(j_b.collect(q).unwrap(), j_t.collect(q).unwrap());
        assert_eq!(j_b.collect(q).unwrap().as_array().unwrap().len(), 2);
    }

    #[test]
    fn deep_find_field_eq_chains_further() {
        
        
        use crate::Jetro;
        let doc = json!({
            "items":[
                {"type":"action","v":10},
                {"type":"idle","v":20},
                {"type":"action","v":30}
            ]
        });
        let raw = serde_json::to_vec(&doc).unwrap();
        let j_b = Jetro::from_bytes(raw).unwrap();
        let j_t = Jetro::new(doc);
        let q = r#"$..find(@.type == "action").map(v)"#;
        assert_eq!(j_b.collect(q).unwrap(), j_t.collect(q).unwrap());
        assert_eq!(j_b.collect(q).unwrap(), json!([10, 30]));
    }

    #[test]
    fn find_shallow_multi_pred_and() {
        
        use crate::Jetro;
        let doc = json!({"xs":[
            {"t":"a","v":1},
            {"t":"a","v":2},
            {"t":"b","v":1}
        ]});
        let j = Jetro::new(doc);
        let r = j.collect(r#"$.xs.find(@.t == "a", @.v == 1)"#).unwrap();
        assert_eq!(r, json!([{"t":"a","v":1}]));
    }

    #[test]
    fn find_shallow_single_pred_still_works() {
        use crate::Jetro;
        let doc = json!({"xs":[{"v":1},{"v":2}]});
        let j = Jetro::new(doc);
        let r = j.collect(r#"$.xs.find(@.v == 2)"#).unwrap();
        assert_eq!(r, json!([{"v":2}]));
    }

    #[test]
    fn deep_find_multi_pred_matches_scan_and_tree() {
        
        use crate::Jetro;
        let doc = json!({
            "rows":[
                {"type":"action","device":"mobile","id":1},
                {"type":"action","device":"desktop","id":2},
                {"type":"idle","device":"mobile","id":3},
                {"nested":{"type":"action","device":"mobile","id":4}}
            ]
        });
        let raw = serde_json::to_vec(&doc).unwrap();
        let j_b = Jetro::from_bytes(raw).unwrap();
        let j_t = Jetro::new(doc);
        let q = r#"$..find(@.type == "action", @.device == "mobile")"#;
        let rb = j_b.collect(q).unwrap();
        let rt = j_t.collect(q).unwrap();
        assert_eq!(rb, rt);
        assert_eq!(rb.as_array().unwrap().len(), 2);
    }

    #[test]
    fn deep_find_then_filter_eq_refines_spans() {
        
        
        use crate::Jetro;
        let doc = json!({
            "rows":[
                {"type":"action","device":"mobile","state":"on","id":1},
                {"type":"action","device":"mobile","state":"off","id":2},
                {"type":"action","device":"desktop","state":"on","id":3}
            ]
        });
        let raw = serde_json::to_vec(&doc).unwrap();
        let j_b = Jetro::from_bytes(raw).unwrap();
        let j_t = Jetro::new(doc);
        let q = r#"$..find(@.type == "action", @.device == "mobile").filter(@.state == "on")"#;
        let rb = j_b.collect(q).unwrap();
        let rt = j_t.collect(q).unwrap();
        assert_eq!(rb, rt);
        assert_eq!(rb.as_array().unwrap().len(), 1);
    }

    #[test]
    fn deep_find_then_filter_cmp_refines_spans() {
        
        use crate::Jetro;
        let doc = json!({
            "rows":[
                {"type":"action","v":1},
                {"type":"action","v":10},
                {"type":"action","v":100},
                {"type":"idle","v":200}
            ]
        });
        let raw = serde_json::to_vec(&doc).unwrap();
        let j_b = Jetro::from_bytes(raw).unwrap();
        let j_t = Jetro::new(doc);
        let q = r#"$..find(@.type == "action").filter(@.v > 5).map(v).sum()"#;
        let rb = j_b.collect(q).unwrap();
        let rt = j_t.collect(q).unwrap();
        assert_eq!(rb, rt);
        assert_eq!(rb, json!(110));
    }

    #[test]
    fn deep_find_then_filter_then_count() {
        
        use crate::Jetro;
        let doc = json!({
            "rows":[
                {"type":"action","v":1},
                {"type":"action","v":10},
                {"type":"action","v":100},
                {"type":"idle","v":50}
            ]
        });
        let raw = serde_json::to_vec(&doc).unwrap();
        let j_b = Jetro::from_bytes(raw).unwrap();
        let q = r#"$..find(@.type == "action").filter(@.v >= 10).count()"#;
        assert_eq!(j_b.collect(q).unwrap(), json!(2));
    }

    #[test]
    fn deep_find_then_fused_filter_map_sum() {
        
        
        use crate::Jetro;
        let doc = json!({
            "rows":[
                {"type":"action","v":1},
                {"type":"action","v":10},
                {"type":"action","v":100}
            ]
        });
        let raw = serde_json::to_vec(&doc).unwrap();
        let j_b = Jetro::from_bytes(raw).unwrap();
        let q = r#"$..find(@.type == "action").filter(@.v > 5).map(v).sum()"#;
        assert_eq!(j_b.collect(q).unwrap(), json!(110));
    }

    #[test]
    fn deep_find_then_fused_filter_map_array() {
        use crate::Jetro;
        let doc = json!({
            "rows":[
                {"type":"action","v":1},
                {"type":"action","v":10},
                {"type":"idle","v":50}
            ]
        });
        let raw = serde_json::to_vec(&doc).unwrap();
        let j_b = Jetro::from_bytes(raw.clone()).unwrap();
        let j_t = Jetro::new(serde_json::from_slice::<serde_json::Value>(&raw).unwrap());
        let q = r#"$..find(@.type == "action").filter(@.v >= 10).map(v)"#;
        assert_eq!(j_b.collect(q).unwrap(), j_t.collect(q).unwrap());
        assert_eq!(j_b.collect(q).unwrap(), json!([10]));
    }

    
    #[test]
    fn route_c_one_mismatch_errors_via_fallthrough() {
        
        
        use crate::Jetro;
        let doc = json!({"xs":[{"id":1},{"id":2}]});
        let raw = serde_json::to_vec(&doc).unwrap();
        let j_b = Jetro::from_bytes(raw).unwrap();
        assert!(j_b.collect("$..id.one()").is_err());
    }
}
