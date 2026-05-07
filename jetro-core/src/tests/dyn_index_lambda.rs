//! Regression: dynamic key indexing inside lambdas. The VM `DynIndex`
//! opcode previously matched only `Val::Str` for string keys and fell
//! through to `Val::Null` for `Val::StrSlice` (the borrowed-string variant
//! produced by simd-json tape reads). Inside a `.map(...)` body, `@.author`
//! reads a `StrSlice`, so `$.posts.map($.realnames[@.author])` returned all
//! nulls when run via `Jetro::from_bytes` (tape-backed) but worked through
//! the legacy `vm_query` path (which used owned `Val::Str`).

#[test]
fn dyn_index_inside_map_via_jetro_bytes() {
    let bytes = br#"{"posts":[{"title":"First","author":"anon"},{"title":"Other","author":"person1"}],"realnames":{"anon":"Anonymous Coward","person1":"Person McPherson"}}"#.to_vec();
    let j = crate::Jetro::from_bytes(bytes).unwrap();
    let v: serde_json::Value = j.collect("$.posts.map($.realnames[@.author])").unwrap();
    assert_eq!(
        v,
        serde_json::json!(["Anonymous Coward", "Person McPherson"]),
    );
}

#[test]
fn dyn_index_inside_filter_via_jetro_bytes() {
    let bytes = br#"{"users":[{"role":"admin","id":"u1"},{"role":"user","id":"u2"}],"perms":{"u1":"all","u2":"read"}}"#.to_vec();
    let j = crate::Jetro::from_bytes(bytes).unwrap();
    let v: serde_json::Value = j
        .collect("$.users.filter($.perms[@.id] == \"all\").map(@.id)")
        .unwrap();
    assert_eq!(v, serde_json::json!(["u1"]));
}
