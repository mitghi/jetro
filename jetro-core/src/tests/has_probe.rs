#[test]
fn test_has_arr_lit() {
    use jetro_core::Jetro;
    let doc = r#"{"obj":{"a":1,"b":2,"c":3,"d":4}}"#;
    let j = Jetro::from_bytes(doc.as_bytes()).unwrap();
    let queries = [
        r#"$.obj.filter_keys(["a","c"] has @)"#,
        r#"$.obj.filter_keys(lambda k: ["a","c"] has k)"#,
        r#"$.obj.filter_keys(@ has "a")"#,
    ];
    for q in &queries {
        let r: Result<serde_json::Value, _> = j.collect(q);
        match r {
            Ok(v) => eprintln!("OK  {} => {}", q, v),
            Err(e) => eprintln!("ERR {} => {:?}", q, e),
        }
    }
}
