//! Coverage for builtin runtime migrations landed in v0.5.5.
//!
//! - `parse_int(radix)` — base-aware integer parsing (2..=36) with optional
//!   `0x` / `0b` / `0o` prefix stripping.
//! - `to_csv(headers)` / `to_tsv(headers)` — explicit-header column ordering
//!   with header line emitted first.
//! - `accumulate(init, fn)` — two-arg fold variant: explicit initial
//!   accumulator. Pre-existing single-arg form (`accumulate(fn)`) preserved.
//! - `rec` — fixpoint detection now uses deep structural equality (was
//!   scalar-only, looping forever on object inputs).

use super::common::vm_query;
use serde_json::json;

// ════════════════════════════════════════════════════════════════════════════
// parse_int(radix)
// ════════════════════════════════════════════════════════════════════════════

mod parse_int_radix {
    use super::*;

    #[test]
    fn no_arg_base10_unchanged() {
        assert_eq!(
            vm_query(r#""42".parse_int()"#, &json!(null)).unwrap(),
            json!(42)
        );
    }

    #[test]
    fn explicit_radix_10() {
        assert_eq!(
            vm_query(r#""42".parse_int(10)"#, &json!(null)).unwrap(),
            json!(42)
        );
    }

    #[test]
    fn hex_no_prefix() {
        assert_eq!(
            vm_query(r#""ff".parse_int(16)"#, &json!(null)).unwrap(),
            json!(255)
        );
    }

    #[test]
    fn hex_with_prefix() {
        assert_eq!(
            vm_query(r#""0xff".parse_int(16)"#, &json!(null)).unwrap(),
            json!(255)
        );
        assert_eq!(
            vm_query(r#""0XFF".parse_int(16)"#, &json!(null)).unwrap(),
            json!(255)
        );
    }

    #[test]
    fn binary_with_prefix() {
        assert_eq!(
            vm_query(r#""0b101".parse_int(2)"#, &json!(null)).unwrap(),
            json!(5)
        );
    }

    #[test]
    fn octal_with_prefix() {
        assert_eq!(
            vm_query(r#""0o17".parse_int(8)"#, &json!(null)).unwrap(),
            json!(15)
        );
    }

    #[test]
    fn base36() {
        assert_eq!(
            vm_query(r#""zz".parse_int(36)"#, &json!(null)).unwrap(),
            json!(35 * 36 + 35)
        );
    }

    #[test]
    fn invalid_returns_null() {
        assert_eq!(
            vm_query(r#""x".parse_int(16)"#, &json!(null)).unwrap(),
            json!(null)
        );
    }

    #[test]
    fn out_of_range_radix_returns_null() {
        assert_eq!(
            vm_query(r#""1".parse_int(40)"#, &json!(null)).unwrap(),
            json!(null)
        );
    }
}

// ════════════════════════════════════════════════════════════════════════════
// to_csv / to_tsv with headers
// ════════════════════════════════════════════════════════════════════════════

mod to_csv_headers {
    use super::*;

    fn records() -> serde_json::Value {
        json!({
            "records": [
                {"id": 1, "name": "a", "email": "x@y.com"},
                {"id": 2, "name": "b", "email": "u@v.com"}
            ]
        })
    }

    #[test]
    fn no_arg_emits_csv() {
        // No-arg form emits cells in object-key iteration order. simd-json
        // tape parsing may not preserve source order, so we test stable
        // properties (right number of rows, right number of cells per row)
        // rather than exact column order.
        let out = vm_query("$.records.to_csv()", &records()).unwrap();
        let s = out.as_str().unwrap();
        assert_eq!(s.lines().count(), 2);
        for line in s.lines() {
            assert_eq!(line.split(',').count(), 3);
        }
    }

    #[test]
    fn explicit_header_order() {
        let out = vm_query(r#"$.records.to_csv(["id", "name", "email"])"#, &records()).unwrap();
        assert_eq!(out, json!("id,name,email\n1,a,x@y.com\n2,b,u@v.com"));
    }

    #[test]
    fn header_subset_reorder() {
        let out = vm_query(r#"$.records.to_csv(["name", "id"])"#, &records()).unwrap();
        assert_eq!(out, json!("name,id\na,1\nb,2"));
    }

    #[test]
    fn missing_header_yields_empty_cell() {
        let out = vm_query(r#"$.records.to_csv(["id", "missing"])"#, &records()).unwrap();
        assert_eq!(out, json!("id,missing\n1,\n2,"));
    }

    #[test]
    fn tsv_with_headers() {
        let out = vm_query(r#"$.records.to_tsv(["id", "email"])"#, &records()).unwrap();
        assert_eq!(out, json!("id\temail\n1\tx@y.com\n2\tu@v.com"));
    }
}

// ════════════════════════════════════════════════════════════════════════════
// accumulate(init, fn)
// ════════════════════════════════════════════════════════════════════════════

mod accumulate_init {
    use super::*;

    #[test]
    fn cumsum_with_init_zero() {
        assert_eq!(
            vm_query("[1,2,3,4,5].accumulate(0, (a, x) => a + x)", &json!(null),).unwrap(),
            json!([1, 3, 6, 10, 15])
        );
    }

    #[test]
    fn cumprod_with_init_one() {
        assert_eq!(
            vm_query("[1,2,3,4].accumulate(1, (a, x) => a * x)", &json!(null),).unwrap(),
            json!([1, 2, 6, 24])
        );
    }

    #[test]
    fn fold_to_string() {
        assert_eq!(
            vm_query(
                r#"[1,2,3].accumulate("", (a, x) => a + x.to_string())"#,
                &json!(null),
            )
            .unwrap(),
            json!(["1", "12", "123"])
        );
    }

    #[test]
    fn empty_input_yields_empty_array() {
        assert_eq!(
            vm_query("[].accumulate(99, (a, x) => a + x)", &json!(null),).unwrap(),
            json!([])
        );
    }

    #[test]
    fn single_arg_form_unchanged() {
        // Existing (no-init) form keeps prior semantics: acc seeded from
        // items[0], output starts at items[0] (not at the seed + items[0]).
        assert_eq!(
            vm_query("[1,2,3,4].accumulate((a, x) => a + x)", &json!(null),).unwrap(),
            json!([1, 3, 6, 10])
        );
    }

    #[test]
    fn path_source_int_vec() {
        let doc = json!({"xs": [10, 20, 30]});
        let out = vm_query("$.xs.accumulate(0, (a, x) => a + x)", &doc).unwrap();
        assert_eq!(out, json!([10, 30, 60]));
    }
}

// ════════════════════════════════════════════════════════════════════════════
// rec fixpoint with deep equality
// ════════════════════════════════════════════════════════════════════════════

mod rec_fixpoint {
    use super::*;

    #[test]
    fn identity_object_converges() {
        // Pre-fix: looped 10000 times because `vals_eq` returned false for
        // any two `Val::Obj`.
        let doc = json!({"o": {"a": 1, "b": 2}});
        let out = vm_query("$.o.rec(d => d)", &doc).unwrap();
        assert_eq!(out, json!({"a": 1, "b": 2}));
    }

    #[test]
    fn idempotent_merge_converges_after_one_iter() {
        let doc = json!({"o": {"type": "v1", "name": "a"}});
        let out = vm_query(r#"$.o.rec(d => d.merge({type: "v2"}))"#, &doc).unwrap();
        assert_eq!(out, json!({"type": "v2", "name": "a"}));
    }

    #[test]
    fn identity_array_converges() {
        let doc = json!({"xs": [1, 2, 3]});
        let out = vm_query("$.xs.rec(a => a)", &doc).unwrap();
        assert_eq!(out, json!([1, 2, 3]));
    }

    #[test]
    fn scalar_bounded_converges() {
        // Numeric input that genuinely needs to iterate.
        let out = vm_query("0.rec(n => n + 1 if n < 10 else n)", &json!(null)).unwrap();
        assert_eq!(out, json!(10));
    }

    #[test]
    fn scalar_identity_converges() {
        assert_eq!(vm_query("42.rec(n => n)", &json!(null)).unwrap(), json!(42));
    }

    #[test]
    fn empty_object_identity_converges() {
        assert_eq!(
            vm_query("({}).rec(d => d)", &json!(null)).unwrap(),
            json!({})
        );
    }

    #[test]
    fn nested_object_identity_converges() {
        let doc = json!({"node": {"type": "v1", "children": [{"type": "v1"}]}});
        let out = vm_query("$.node.rec(d => d)", &doc).unwrap();
        assert_eq!(out, json!({"type": "v1", "children": [{"type": "v1"}]}));
    }
}
