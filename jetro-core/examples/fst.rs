use jetro_core::Jetro;
fn main() {
    let j = Jetro::from_bytes(br#"{"x":[{"id":1,"total":99.0,"customer":{"name":"A","tier":"g"},"items":[{"qty":1,"price":99.0}]}]}"#.to_vec()).unwrap();
    println!(
        "d: {:?}",
        j.collect(r#"$.x.map({a: id, b: total, lbl: f"{id}-{total}"})"#)
    );
    println!(
        "e: {:?}",
        j.collect(r#"$.x.map({a: id, b: total, c: customer.name, lbl: f"{id}-{total}"})"#)
    );
    println!("f: {:?}", j.collect(r#"$.x.map({a: id, b: total, c: customer.name, d: items.map(qty*price).sum(), lbl: f"{id}-{total}"})"#));
}
