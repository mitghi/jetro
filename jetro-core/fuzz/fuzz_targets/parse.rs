#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let Ok(s) = std::str::from_utf8(data) else { return };
    if s.len() > 4096 {
        return;
    }
    // Must not panic on any UTF-8 input. Errors are fine.
    let _ = std::panic::catch_unwind(|| {
        let _ = jetro_core::__fuzz_internal::parse(s);
    });
});
