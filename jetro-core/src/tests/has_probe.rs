#[test]
fn test_has_arr_lit() {
    use crate::Jetro;
    let doc = r#"{"obj":{"a":1,"b":2,"c":3,"d":4}}"#;
    let j = Jetro::from_bytes(doc.as_bytes().to_vec()).unwrap();
    let cases = [
        (
            r#"$.obj.filter_keys(["a","c"] has @)"#,
            serde_json::json!({"a": 1, "c": 3}),
        ),
        (
            r#"$.obj.filter_keys(lambda k: ["a","c"] has k)"#,
            serde_json::json!({"a": 1, "c": 3}),
        ),
        (
            r#"$.obj.filter_keys(@ has "a")"#,
            serde_json::json!({"a": 1}),
        ),
    ];
    for (query, expected) in cases {
        let got: serde_json::Value = j.collect(query).unwrap();
        assert_eq!(got, expected, "{query}");
    }
}
