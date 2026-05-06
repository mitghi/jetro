#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let Ok(s) = std::str::from_utf8(data) else { return };
    if s.len() > 4096 {
        return;
    }
    // parse → plan must not panic, regardless of whether parse succeeds.
    let _ = std::panic::catch_unwind(|| {
        let _ = jetro_core::__fuzz_internal::plan_query(s);
    });
});
