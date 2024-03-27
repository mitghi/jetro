#[cfg(feature = "dhat-heap")]
#[global_allocator]
static ALLOC: dhat::Alloc = dhat::Alloc;

use jetro;

fn tst() {
    let v = serde_json::json!({"some": "value", "another": {"nested": [{"v": "test"}]}});
    let result = jetro::context::Path::collect(v, ">/..v");
}

fn main() {
    #[cfg(feature = "dhat-heap")]
    let _profiler = dhat::Profiler::new_heap();
    for i in 0..100 {
        tst();
    }
}
