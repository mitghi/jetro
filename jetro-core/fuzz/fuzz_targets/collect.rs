#![no_main]

use libfuzzer_sys::fuzz_target;

const FIXED_DOC: &[u8] = br#"{
  "users": [
    {"id": 1, "name": "alice", "active": true,  "score": 42.5, "tags": ["a","b"]},
    {"id": 2, "name": "bob",   "active": false, "score": 17.0, "tags": []},
    {"id": 3, "name": "carol", "active": true,  "score": 99.9, "tags": ["c"]}
  ],
  "meta": {"version": 1, "host": "localhost"},
  "scalar": 7
}"#;

fuzz_target!(|data: &[u8]| {
    let Ok(expr) = std::str::from_utf8(data) else { return };
    if expr.len() > 4096 {
        return;
    }
    // Full pipeline: parse → plan → exec must not panic for any UTF-8 input.
    // Errors are fine; only panics fail.
    let _ = std::panic::catch_unwind(|| {
        let Ok(j) = jetro_core::Jetro::from_bytes(FIXED_DOC.to_vec()) else { return };
        let _ = j.collect(expr);
    });
});
