

#[cfg(test)]
mod examples {
    use crate::{parser, vm, vm::VM};
    use serde_json::{json, Value};

    fn vm_query(expr: &str, doc: &Value) -> Result<Value, crate::Error> {
        let ast = parser::parse(expr)?;
        let program = vm::Compiler::compile(&ast, expr);
        let mut vm = vm::VM::new();
        Ok(vm.execute(&program, doc)?)
    }

    
    fn world() -> Value {
        json!({
            "users": [
                {"id": 1, "name": "Alice", "email": "alice@example.com",
                 "role": "admin", "active": true,  "age": 30, "score": 95,
                 "tags": ["vip", "early-adopter"]},
                {"id": 2, "name": "Bob",   "email": "bob@example.com",
                 "role": "user",  "active": true,  "age": 25, "score": 72,
                 "tags": ["user"]},
                {"id": 3, "name": "Carol", "email": "carol@example.com",
                 "role": "user",  "active": false, "age": 35, "score": 88,
                 "tags": ["vip"]},
                {"id": 4, "name": "Dave",  "email": "dave@example.com",
                 "role": "mod",   "active": true,  "age": 28, "score": 61,
                 "tags": ["mod"]}
            ],
            "products": [
                {"id": "p1", "name": "Widget A", "price": 9.99,
                 "category": "widgets", "tags": ["sale","popular"],
                 "stock": 150, "meta": {"weight": 0.5, "color": "red"}},
                {"id": "p2", "name": "Widget B", "price": 24.99,
                 "category": "widgets", "tags": ["new"],
                 "stock": 30,  "meta": {"weight": 1.2, "color": "blue"}},
                {"id": "p3", "name": "Gadget X", "price": 49.99,
                 "category": "gadgets", "tags": ["popular","featured"],
                 "stock": 0,   "meta": {"weight": 0.3, "color": "black"}},
                {"id": "p4", "name": "Gadget Y", "price": 14.50,
                 "category": "gadgets", "tags": ["sale"],
                 "stock": 75,  "meta": {"weight": 0.8, "color": "white"}}
            ],
            "orders": [
                {"id": "o1", "user_id": 1,
                 "items": [{"product_id": "p1", "qty": 2}, {"product_id": "p3", "qty": 1}],
                 "total": 69.97, "status": "shipped"},
                {"id": "o2", "user_id": 2,
                 "items": [{"product_id": "p2", "qty": 1}],
                 "total": 24.99, "status": "pending"},
                {"id": "o3", "user_id": 1,
                 "items": [{"product_id": "p4", "qty": 3}],
                 "total": 43.50, "status": "delivered"},
                {"id": "o4", "user_id": 3,
                 "items": [{"product_id": "p1", "qty": 1}],
                 "total": 9.99, "status": "pending"}
            ],
            "events": [
                {"etype": "login",    "user_id": 1,    "ts": 1700000000,
                 "data": {"ip": "1.2.3.4"},       "error": null},
                {"etype": "purchase", "user_id": 2,   "ts": 1700001000,
                 "data": {"order_id": "o2"},       "error": null},
                {"etype": "error",    "user_id": null, "ts": 1700002000,
                 "data": {},                        "error": "timeout"}
            ],
            "config": {
                "app": {"name": "Jetro Demo", "version": "2.0", "debug": false},
                "limits": {"max_users": 1000, "max_orders": 50000},
                "flags": {"new_ui": true, "dark_mode": false, "beta_api": true}
            },
            "strings": {
                "padded":    "  Hello, World!  ",
                "slug":      "hello-world",
                "csv":       "name,age\nAlice,30\nBob,25",
                "b64":       "aGVsbG8gd29ybGQ=",
                "html":      "<h1>Hello &amp; World</h1>",
                "url_raw":   "hello world & foo=bar",
                "multiline": "  line one\n  line two\n  line three"
            },
            "numbers": {
                "ints":   [3, 1, 4, 1, 5, 9, 2, 6, 5, 3],
                "floats": [1.1, 2.2, 3.3, 4.4],
                "sparse": [1, null, 2, null, 3]
            },
            "nested": {
                "a": {
                    "b": {"c": {"value": 42, "label": "deep"}, "d": 10},
                    "e": [1, 2, 3]
                }
            },
            "flat": {
                "user.name":   "Alice",
                "user.age":    30,
                "config.debug": false
            },
            "sets": {
                "alpha": [1, 2, 3, 4, 5],
                "beta":  [3, 4, 5, 6, 7]
            },
            "pivot_data": [
                {"region": "north", "product": "A", "sales": 100},
                {"region": "north", "product": "B", "sales": 200},
                {"region": "south", "product": "A", "sales": 150}
            ],
            "mixed_types": [
                {"v": 1},
                {"v": "hello"},
                {"v": true},
                {"v": null},
                {"v": [1, 2]},
                {"v": {"x": 1}}
            ]
        })
    }

    fn q(expr: &str) -> Value {
        vm_query(expr, &world()).expect(expr)
    }

    
    #[test]
    fn nav_field() {
        
        assert_eq!(q("$.config.app.name"), json!("Jetro Demo"));
    }

    #[test]
    fn nav_index_positive() {
        assert_eq!(q("$.users[0].name"), json!("Alice"));
    }

    #[test]
    fn nav_index_negative() {
        
        assert_eq!(q("$.users[-1].name"), json!("Dave"));
    }

    #[test]
    fn nav_slice() {
        
        let r = q("$.users[1:3].map(name)");
        assert_eq!(r, json!(["Bob", "Carol"]));
    }

    #[test]
    fn nav_slice_open_end() {
        let r = q("$.users[2:].map(name)");
        assert_eq!(r, json!(["Carol", "Dave"]));
    }

    #[test]
    fn nav_descendant() {
        
        let r = q("$..color");
        let arr = r.as_array().unwrap();
        assert!(arr.contains(&json!("red")));
        assert!(arr.contains(&json!("black")));
        assert_eq!(arr.len(), 4);
    }

    #[test]
    fn nav_optional_field_null_safe() {
        
        let doc = json!({"user": null});
        assert_eq!(vm_query("$.user?.name", &doc).unwrap(), json!(null));
    }

    #[test]
    fn nav_optional_chain() {
        let doc = json!({"users": [{"id": 1}, {"id": 2, "profile": {"bio": "hi"}}]});
        
        let r = vm_query("$.users[0].profile?.bio", &doc).unwrap();
        assert_eq!(r, json!(null));
        let r2 = vm_query("$.users[1].profile?.bio", &doc).unwrap();
        assert_eq!(r2, json!("hi"));
    }

    
    #[test]
    fn filter_gt() {
        let r = q("$.users.filter(score > 80).map(name)");
        assert_eq!(r, json!(["Alice", "Carol"]));
    }

    #[test]
    fn filter_and() {
        let r = q("$.users.filter(active == true and score >= 90).map(name)");
        assert_eq!(r, json!(["Alice"]));
    }

    #[test]
    fn filter_or() {
        let r = q("$.users.filter(role == \"admin\" or role == \"mod\").map(name)");
        assert_eq!(r, json!(["Alice", "Dave"]));
    }

    #[test]
    fn filter_not() {
        let r = q("$.users.filter(not active).map(name)");
        assert_eq!(r, json!(["Carol"]));
    }

    #[test]
    fn filter_lambda() {
        let r = q("$.products.filter(lambda p: p.stock == 0).map(name)");
        assert_eq!(r, json!(["Gadget X"]));
    }

    #[test]
    fn filter_fuzzy() {
        
        let r = q("$.products.filter(name ~= \"widget\").map(id)");
        let arr = r.as_array().unwrap();
        assert_eq!(arr.len(), 2);
    }

    #[test]
    fn filter_includes() {
        
        let r = q("$.products.filter(tags.includes(\"sale\")).map(name)");
        let arr = r.as_array().unwrap();
        assert!(arr.contains(&json!("Widget A")));
        assert!(arr.contains(&json!("Gadget Y")));
    }

    #[test]
    fn filter_kind_number() {
        
        let r = q("$.mixed_types.filter(v kind number)");
        assert_eq!(r.as_array().unwrap().len(), 1);
    }

    #[test]
    fn filter_kind_not_null() {
        let r = q("$.events.filter(user_id kind not null).map(etype)");
        assert_eq!(r, json!(["login", "purchase"]));
    }

    #[test]
    fn filter_kind_string() {
        let r = q("$.mixed_types.filter(v kind string)");
        assert_eq!(r.as_array().unwrap().len(), 1);
    }

    #[test]
    fn filter_kind_object() {
        let r = q("$.mixed_types.filter(v kind object)");
        assert_eq!(r.as_array().unwrap().len(), 1);
    }

    #[test]
    fn filter_kind_array() {
        let r = q("$.mixed_types.filter(v kind array)");
        assert_eq!(r.as_array().unwrap().len(), 1);
    }

    #[test]
    fn filter_kind_bool() {
        let r = q("$.mixed_types.filter(v kind bool)");
        assert_eq!(r.as_array().unwrap().len(), 1);
    }

    
    #[test]
    fn map_pluck_field() {
        assert_eq!(
            q("$.users.map(name)"),
            json!(["Alice", "Bob", "Carol", "Dave"])
        );
    }

    #[test]
    fn map_object_shorthand() {
        let r = q("$.users.map({name, role})");
        let arr = r.as_array().unwrap();
        assert_eq!(arr[0]["role"], json!("admin"));
    }

    #[test]
    fn map_rename() {
        let r = q("$.products.map({id, title: name, cost: price})");
        let arr = r.as_array().unwrap();
        assert_eq!(arr[0]["title"], json!("Widget A"));
        assert_eq!(arr[0]["cost"], json!(9.99));
        assert!(arr[0].get("name").is_none());
    }

    #[test]
    fn map_computed() {
        let r = q("$.products.map({name, in_stock: stock > 0})");
        let arr = r.as_array().unwrap();
        assert_eq!(arr[2]["in_stock"], json!(false)); 
        assert_eq!(arr[0]["in_stock"], json!(true));
    }

    #[test]
    fn map_lambda() {
        let r = q("$.numbers.ints.map(lambda n: n * n)");
        let arr = r.as_array().unwrap();
        assert_eq!(arr[0], json!(9)); 
    }

    #[test]
    fn map_nested_field() {
        
        let r = q("$.products.map({name, color: meta.color})");
        let arr = r.as_array().unwrap();
        assert_eq!(arr[0]["color"], json!("red"));
    }

    #[test]
    fn flat_map_basic() {
        
        let r = q("$.orders.flat_map(items)");
        let arr = r.as_array().unwrap();
        
        assert_eq!(arr.len(), 5);
    }

    
    #[test]
    fn agg_len() {
        assert_eq!(q("$.users.len()"), json!(4));
    }

    #[test]
    fn agg_sum_direct() {
        assert_eq!(q("$.numbers.ints.sum()"), json!(39));
    }

    #[test]
    fn agg_sum_field() {
        let total = q("$.orders.sum(total)");
        let v = total.as_f64().unwrap();
        assert!((v - 148.45).abs() < 0.01);
    }

    #[test]
    fn agg_avg_field() {
        let r = q("$.users.avg(score)");
        let v = r.as_f64().unwrap();
        
        assert!((v - 79.0).abs() < 0.1);
    }

    #[test]
    fn agg_min_max_field() {
        assert_eq!(q("$.users.min(score)"), json!(61));
        assert_eq!(q("$.users.max(score)"), json!(95));
    }

    #[test]
    fn agg_count_with_predicate() {
        
        assert_eq!(q("$.users.count(active == true)"), json!(3));
    }

    #[test]
    fn agg_count_no_args() {
        assert_eq!(q("$.orders.count()"), json!(4));
    }

    #[test]
    fn agg_any_all() {
        assert_eq!(q("$.users.any(active == true)"), json!(true));
        assert_eq!(q("$.users.all(active == true)"), json!(false));
        assert_eq!(q("$.users.all(score > 0)"), json!(true));
    }

    #[test]
    fn agg_group_by() {
        let r = q("$.users.group_by(role)");
        let obj = r.as_object().unwrap();
        assert_eq!(obj["admin"].as_array().unwrap().len(), 1);
        assert_eq!(obj["user"].as_array().unwrap().len(), 2);
    }

    #[test]
    fn agg_count_by() {
        let r = q("$.orders.count_by(status)");
        assert_eq!(r["pending"], json!(2));
        assert_eq!(r["shipped"], json!(1));
    }

    #[test]
    fn agg_index_by() {
        
        let r = q("$.users.index_by(id)");
        let obj = r.as_object().unwrap();
        assert_eq!(obj.len(), 4);
        assert_eq!(obj["1"]["name"], json!("Alice"));
    }

    
    #[test]
    fn arr_sort_asc() {
        let r = q("$.users.sort(score).map(name)");
        let arr = r.as_array().unwrap();
        assert_eq!(arr[0], json!("Dave")); 
    }

    #[test]
    fn arr_sort_desc() {
        let r = q("$.users.sort(-score).map(name)");
        assert_eq!(r.as_array().unwrap()[0], json!("Alice")); 
    }

    #[test]
    fn arr_sort_lambda() {
        let r = q("$.products.sort(lambda a, b: a.price < b.price).map(id)");
        assert_eq!(r.as_array().unwrap()[0], json!("p1"));
    }

    #[test]
    fn arr_reverse() {
        let r = q("$.numbers.ints.reverse()");
        let arr = r.as_array().unwrap();
        assert_eq!(arr[0], json!(3)); 
    }

    #[test]
    fn arr_unique() {
        let r = q("$.numbers.ints.unique()");
        let arr = r.as_array().unwrap();
        
        assert_eq!(arr.len(), 7);
    }

    #[test]
    fn arr_flatten() {
        let doc = json!({"x": [[1, 2], [3, [4, 5]]]});
        let r = vm_query("$.x.flatten()", &doc).unwrap();
        assert_eq!(r, json!([1, 2, 3, [4, 5]]));
    }

    #[test]
    fn arr_flatten_deep() {
        let doc = json!({"x": [[1, [2, [3]]]]});
        let r = vm_query("$.x.flatten(10)", &doc).unwrap();
        assert_eq!(r, json!([1, 2, 3]));
    }

    #[test]
    fn arr_compact() {
        let r = q("$.numbers.sparse.compact()");
        assert_eq!(r, json!([1, 2, 3]));
    }

    #[test]
    fn arr_first_last() {
        assert_eq!(q("$.users.first().name"), json!("Alice"));
        assert_eq!(q("$.users.last().name"), json!("Dave"));
    }

    #[test]
    fn arr_first_n() {
        let r = q("$.users.first(2).map(name)");
        assert_eq!(r, json!(["Alice", "Bob"]));
    }

    #[test]
    fn arr_last_n() {
        let r = q("$.users.last(2).map(name)");
        assert_eq!(r, json!(["Carol", "Dave"]));
    }

    #[test]
    fn arr_nth() {
        
        assert_eq!(q("$.users.nth(2).name"), json!("Carol"));
    }

    #[test]
    fn arr_append_prepend() {
        let doc = json!({"vals": [2, 3]});
        assert_eq!(
            vm_query("$.vals.append(4)", &doc).unwrap(),
            json!([2, 3, 4])
        );
        assert_eq!(
            vm_query("$.vals.prepend(1)", &doc).unwrap(),
            json!([1, 2, 3])
        );
    }

    #[test]
    fn arr_remove_by_predicate() {
        let doc = json!({"vals": [1, 2, 3, 4, 5]});
        let r = vm_query("$.vals.remove(lambda v: v % 2 == 0)", &doc).unwrap();
        assert_eq!(r, json!([1, 3, 5]));
    }

    #[test]
    fn arr_join() {
        let doc = json!({"words": ["hello", "world"]});
        assert_eq!(
            vm_query("$.words.join(\", \")", &doc).unwrap(),
            json!("hello, world")
        );
    }

    
    #[test]
    fn iter_enumerate() {
        let r = q("$.products[0:2].enumerate()");
        let arr = r.as_array().unwrap();
        assert_eq!(arr[0]["index"], json!(0));
        assert_eq!(arr[1]["index"], json!(1));
        assert_eq!(arr[0]["value"]["id"], json!("p1"));
    }

    #[test]
    fn iter_pairwise() {
        let doc = json!({"v": [1, 2, 3, 4]});
        let r = vm_query("$.v.pairwise()", &doc).unwrap();
        assert_eq!(r, json!([[1, 2], [2, 3], [3, 4]]));
    }

    #[test]
    fn iter_window() {
        let doc = json!({"v": [1, 2, 3, 4, 5]});
        let r = vm_query("$.v.window(3)", &doc).unwrap();
        assert_eq!(r, json!([[1, 2, 3], [2, 3, 4], [3, 4, 5]]));
    }

    #[test]
    fn iter_chunk() {
        let doc = json!({"v": [1, 2, 3, 4, 5]});
        let r = vm_query("$.v.chunk(2)", &doc).unwrap();
        assert_eq!(r, json!([[1, 2], [3, 4], [5]]));
    }

    #[test]
    fn iter_takewhile_dropwhile() {
        let doc = json!({"v": [1, 2, 3, 4, 5]});
        assert_eq!(
            vm_query("$.v.takewhile(lambda x: x < 4)", &doc).unwrap(),
            json!([1, 2, 3])
        );
        assert_eq!(
            vm_query("$.v.dropwhile(lambda x: x < 3)", &doc).unwrap(),
            json!([3, 4, 5])
        );
    }

    #[test]
    fn iter_accumulate() {
        let doc = json!({"v": [1, 2, 3, 4]});
        let r = vm_query("$.v.accumulate(lambda acc, x: acc + x)", &doc).unwrap();
        assert_eq!(r, json!([1, 3, 6, 10]));
    }

    #[test]
    fn iter_partition() {
        let doc = json!({"v": [1, 2, 3, 4, 5, 6]});
        let r = vm_query("$.v.partition(lambda n: n % 2 == 0)", &doc).unwrap();
        assert_eq!(r["true"], json!([2, 4, 6]));
        assert_eq!(r["false"], json!([1, 3, 5]));
    }

    #[test]
    fn iter_zip_method() {
        let doc = json!({"a": [1,2,3], "b": ["x","y","z"]});
        let r = vm_query("$.a.zip($.b)", &doc).unwrap();
        assert_eq!(r, json!([[1, "x"], [2, "y"], [3, "z"]]));
    }

    #[test]
    fn iter_zip_longest() {
        let doc = json!({"a": [1,2,3], "b": ["x","y"]});
        let r = vm_query("$.a.zip_longest($.b)", &doc).unwrap();
        assert_eq!(r.as_array().unwrap().len(), 3);
        assert_eq!(r[2][1], json!(null)); 
    }

    
    #[test]
    fn set_diff() {
        let r = q("$.sets.alpha.diff($.sets.beta)");
        assert_eq!(r, json!([1, 2]));
    }

    #[test]
    fn set_intersect() {
        let r = q("$.sets.alpha.intersect($.sets.beta)");
        assert_eq!(r, json!([3, 4, 5]));
    }

    #[test]
    fn set_union() {
        let r = q("$.sets.alpha.union($.sets.beta)");
        let arr = r.as_array().unwrap();
        assert_eq!(arr.len(), 7);
    }

    
    #[test]
    fn obj_keys_values_entries() {
        let r = q("$.config.flags.keys()");
        let keys = r.as_array().unwrap();
        assert!(keys.contains(&json!("new_ui")));

        let v = q("$.config.flags.values()");
        assert!(v.as_array().unwrap().contains(&json!(true)));

        let e = q("$.config.flags.entries()");
        let entries = e.as_array().unwrap();
        
        assert!(entries.iter().any(|e| e[0] == json!("dark_mode")));
    }

    #[test]
    fn obj_pick() {
        let r = q("$.users[0].pick(\"name\", \"role\")");
        assert_eq!(r["name"], json!("Alice"));
        assert_eq!(r["role"], json!("admin"));
        assert!(r.get("score").is_none());
    }

    #[test]
    fn obj_omit() {
        let r = q("$.users[0].omit(\"email\", \"tags\")");
        assert!(r.get("email").is_none());
        assert_eq!(r["name"], json!("Alice"));
    }

    #[test]
    fn obj_merge() {
        
        let doc = json!({"a": {"x": 1, "y": 2}, "b": {"y": 99, "z": 3}});
        let r = vm_query("$.a | merge($.b)", &doc).unwrap();
        assert_eq!(r["y"], json!(99));
        assert_eq!(r["z"], json!(3));
    }

    #[test]
    fn obj_deep_merge() {
        let doc = json!({"a": {"x": {"p": 1}}, "b": {"x": {"q": 2}, "y": 3}});
        let r = vm_query("$.a | deep_merge($.b)", &doc).unwrap();
        assert_eq!(r["x"]["p"], json!(1));
        assert_eq!(r["x"]["q"], json!(2));
    }

    #[test]
    fn obj_defaults() {
        let doc = json!({"obj": {"a": 1, "b": null}, "defs": {"b": 99, "c": 100}});
        let r = vm_query("$.obj.defaults($.defs)", &doc).unwrap();
        assert_eq!(r["b"], json!(99));
        assert_eq!(r["c"], json!(100));
        assert_eq!(r["a"], json!(1));
    }

    #[test]
    fn obj_rename() {
        let doc = json!({"obj": {"old_key": "value", "keep": 1}});
        let r = vm_query("$.obj.rename({old_key: \"new_key\"})", &doc).unwrap();
        assert!(r.get("old_key").is_none());
        assert_eq!(r["new_key"], json!("value"));
    }

    #[test]
    fn obj_transform_keys() {
        let r = q("$.config.flags.transform_keys(lambda k: k.upper())");
        assert!(r.get("NEW_UI").is_some());
        assert!(r.get("DARK_MODE").is_some());
    }

    #[test]
    fn obj_transform_values() {
        let doc = json!({"m": {"a": 1, "b": 2, "c": 3}});
        let r = vm_query("$.m.transform_values(lambda v: v * 10)", &doc).unwrap();
        assert_eq!(r["a"], json!(10));
        assert_eq!(r["b"], json!(20));
    }

    #[test]
    fn obj_filter_keys() {
        let r = q("$.users[0].filter_keys(lambda k: not k.starts_with(\"e\"))");
        assert!(r.get("email").is_none());
        assert!(r.get("name").is_some());
    }

    #[test]
    fn obj_filter_values() {
        let r = q("$.config.flags.filter_values(lambda v: v == true)");
        let obj = r.as_object().unwrap();
        assert!(obj.contains_key("new_ui"));
        assert!(obj.contains_key("beta_api"));
        assert!(!obj.contains_key("dark_mode"));
    }

    #[test]
    fn obj_invert() {
        let doc = json!({"m": {"a": "x", "b": "y"}});
        let r = vm_query("$.m.invert()", &doc).unwrap();
        assert_eq!(r["x"], json!("a"));
    }

    #[test]
    fn obj_to_pairs_from_pairs() {
        let r = q("$.config.flags.to_pairs()");
        let arr = r.as_array().unwrap();
        assert_eq!(arr.len(), 3);

        let doc = json!({"pairs": r});
        let restored = vm_query("$.pairs.from_pairs()", &doc).unwrap();
        assert_eq!(restored["new_ui"], json!(true));
    }

    #[test]
    fn obj_pivot() {
        
        let r = q("$.pivot_data.pivot(\"region\", \"product\", \"sales\")");
        let obj = r.as_object().unwrap();
        assert_eq!(obj["north"]["A"], json!(100));
        assert_eq!(obj["south"]["A"], json!(150));
    }

    
    #[test]
    fn path_get() {
        assert_eq!(q("$.nested.get_path(\"a.b.c.value\")"), json!(42));
    }

    #[test]
    fn path_set() {
        let r = q("$.nested.set_path(\"a.b.d\", 999)");
        assert_eq!(r["a"]["b"]["d"], json!(999));
        assert_eq!(r["a"]["b"]["c"]["value"], json!(42)); 
    }

    #[test]
    fn path_del() {
        let r = q("$.nested.del_path(\"a.b.d\")");
        assert!(r["a"]["b"].get("d").is_none());
        assert!(r["a"]["b"].get("c").is_some());
    }

    #[test]
    fn path_has() {
        assert_eq!(q("$.nested.has_path(\"a.b.c.value\")"), json!(true));
        assert_eq!(q("$.nested.has_path(\"a.b.z\")"), json!(false));
    }

    #[test]
    fn path_flatten_keys() {
        let r = q("$.nested.flatten_keys()");
        assert_eq!(r["a.b.c.value"], json!(42));
        assert_eq!(r["a.b.d"], json!(10));
    }

    #[test]
    fn path_unflatten_keys() {
        let r = q("$.flat.unflatten_keys()");
        assert_eq!(r["user"]["name"], json!("Alice"));
        assert_eq!(r["user"]["age"], json!(30));
        assert_eq!(r["config"]["debug"], json!(false));
    }

    
    #[test]
    fn str_case() {
        let s = "$.strings.padded";
        assert_eq!(q(&format!("{s}.trim().upper()")), json!("HELLO, WORLD!"));
        assert_eq!(q(&format!("{s}.trim().lower()")), json!("hello, world!"));
        assert_eq!(
            q(&format!("{s}.trim().capitalize()")),
            json!("Hello, world!")
        );
        assert_eq!(
            q(&format!("{s}.trim().title_case()")),
            json!("Hello, World!")
        );
    }

    #[test]
    fn str_trim_variants() {
        assert_eq!(q("$.strings.padded.trim()"), json!("Hello, World!"));
        assert_eq!(q("$.strings.padded.trim_left()"), json!("Hello, World!  "));
        assert_eq!(q("$.strings.padded.trim_right()"), json!("  Hello, World!"));
    }

    #[test]
    fn str_split_join() {
        let doc = json!({"s": "a,b,c"});
        let split = vm_query("$.s.split(\",\")", &doc).unwrap();
        assert_eq!(split, json!(["a", "b", "c"]));
        let joined = vm_query("$.s.split(\",\").join(\" | \")", &doc).unwrap();
        assert_eq!(joined, json!("a | b | c"));
    }

    #[test]
    fn str_replace_all() {
        let doc = json!({"s": "foo foo foo"});
        assert_eq!(
            vm_query("$.s.replace(\"foo\", \"bar\")", &doc).unwrap(),
            json!("bar foo foo")
        );
        assert_eq!(
            vm_query("$.s.replace_all(\"foo\", \"bar\")", &doc).unwrap(),
            json!("bar bar bar")
        );
    }

    #[test]
    fn str_starts_ends_strip() {
        let doc = json!({"s": "foobar"});
        assert_eq!(
            vm_query("$.s.starts_with(\"foo\")", &doc).unwrap(),
            json!(true)
        );
        assert_eq!(
            vm_query("$.s.ends_with(\"bar\")", &doc).unwrap(),
            json!(true)
        );
        assert_eq!(
            vm_query("$.s.strip_prefix(\"foo\")", &doc).unwrap(),
            json!("bar")
        );
        assert_eq!(
            vm_query("$.s.strip_suffix(\"bar\")", &doc).unwrap(),
            json!("foo")
        );
    }

    #[test]
    fn str_pad_repeat() {
        let doc = json!({"s": "hi"});
        assert_eq!(
            vm_query("$.s.pad_left(5, \"0\")", &doc).unwrap(),
            json!("000hi")
        );
        assert_eq!(
            vm_query("$.s.pad_right(5, \".\")", &doc).unwrap(),
            json!("hi...")
        );
        assert_eq!(vm_query("$.s.repeat(3)", &doc).unwrap(), json!("hihihi"));
    }

    #[test]
    fn str_index_slice() {
        let doc = json!({"s": "hello world"});
        assert_eq!(vm_query("$.s.index_of(\"world\")", &doc).unwrap(), json!(6));
        assert_eq!(
            vm_query("$.s.last_index_of(\"l\")", &doc).unwrap(),
            json!(9)
        );
        assert_eq!(vm_query("$.s.slice(6, 11)", &doc).unwrap(), json!("world"));
        assert_eq!(vm_query("$.s.slice(0, 5)", &doc).unwrap(), json!("hello"));
    }

    #[test]
    fn str_lines_words_chars() {
        let doc = json!({"s": "a b\nc d"});
        assert_eq!(
            vm_query("$.s.lines()", &doc).unwrap(),
            json!(["a b", "c d"])
        );
        assert_eq!(
            vm_query("$.s.words()", &doc).unwrap(),
            json!(["a", "b", "c", "d"])
        );
        assert_eq!(vm_query("$.s.chars().len()", &doc).unwrap(), json!(7));
    }

    #[test]
    fn str_indent_dedent() {
        let doc = json!({"s": "line one\nline two"});
        let indented = vm_query("$.s.indent(4)", &doc).unwrap();
        assert!(indented.as_str().unwrap().starts_with("    line"));
        let dedented = vm_query("$.strings.multiline.dedent()", &world()).unwrap();
        assert!(dedented.as_str().unwrap().starts_with("line"));
    }

    #[test]
    fn str_matches_scan() {
        let doc = json!({"s": "hello world"});
        assert_eq!(
            vm_query("$.s.matches(\"world\")", &doc).unwrap(),
            json!(true)
        );
        assert_eq!(
            vm_query("$.s.matches(\"xyz\")", &doc).unwrap(),
            json!(false)
        );
        let r = vm_query("$.s.scan(\"l\")", &doc).unwrap();
        
        assert_eq!(r.as_array().unwrap().len(), 3);
    }

    #[test]
    fn str_to_number_to_bool() {
        let doc = json!({"n": "42", "f": "3.14", "b": "true"});
        assert_eq!(vm_query("$.n.to_number()", &doc).unwrap(), json!(42));
        assert_eq!(vm_query("$.f.to_number()", &doc).unwrap(), json!(3.14));
        assert_eq!(vm_query("$.b.to_bool()", &doc).unwrap(), json!(true));
    }

    #[test]
    fn str_base64() {
        
        assert_eq!(q("$.strings.b64.from_base64()"), json!("hello world"));
        let doc = json!({"s": "hello world"});
        assert_eq!(
            vm_query("$.s.to_base64()", &doc).unwrap(),
            json!("aGVsbG8gd29ybGQ=")
        );
    }

    #[test]
    fn str_url_encode_decode() {
        let doc = json!({"s": "hello world"});
        let enc = vm_query("$.s.url_encode()", &doc).unwrap();
        let enc_doc = json!({"s": enc});
        let dec = vm_query("$.s.url_decode()", &enc_doc).unwrap();
        assert_eq!(dec, json!("hello world"));
    }

    #[test]
    fn str_html_unescape() {
        
        let r = q("$.strings.html.html_unescape()");
        assert_eq!(r, json!("<h1>Hello & World</h1>"));
    }

    #[test]
    fn str_to_string_to_json_from_json() {
        let doc = json!({"n": 42, "s": "{\"x\":1}"});
        assert_eq!(vm_query("$.n.to_string()", &doc).unwrap(), json!("42"));
        let parsed = vm_query("$.s.from_json()", &doc).unwrap();
        assert_eq!(parsed["x"], json!(1));
        let round = vm_query("$.s.from_json().to_json()", &doc).unwrap();
        assert!(round.as_str().unwrap().contains("\"x\""));
    }

    
    #[test]
    fn type_method() {
        assert_eq!(q("$.numbers.ints[0].type()"), json!("number"));
        assert_eq!(q("$.strings.slug.type()"), json!("string"));
        assert_eq!(q("$.users.type()"), json!("array"));
        assert_eq!(q("$.config.type()"), json!("object"));
        assert_eq!(q("$.config.flags.new_ui.type()"), json!("bool"));
        assert_eq!(q("$.events[0].error.type()"), json!("null"));
    }

    
    #[test]
    fn null_or_default() {
        let doc = json!({"user": {"name": "Alice", "phone": null}});
        assert_eq!(
            vm_query("$.user.phone.or(\"n/a\")", &doc).unwrap(),
            json!("n/a")
        );
        assert_eq!(
            vm_query("$.user.name.or(\"n/a\")", &doc).unwrap(),
            json!("Alice")
        );
    }

    #[test]
    fn null_has_missing() {
        assert_eq!(q("$.users[0].has(\"email\")"), json!(true));
        assert_eq!(q("$.users[0].has(\"phone\")"), json!(false));
        assert_eq!(q("$.users[0].missing(\"phone\")"), json!(true));
    }

    #[test]
    fn null_coalesce_operator() {
        
        let doc = json!({"a": null, "b": null, "c": 42});
        assert_eq!(vm_query("$.a ?| $.b ?| $.c", &doc).unwrap(), json!(42));
        assert_eq!(vm_query("$.c ?| $.a", &doc).unwrap(), json!(42));
    }

    #[test]
    fn null_compact_pipeline() {
        let r = q("$.numbers.sparse.compact().sum()");
        assert_eq!(r, json!(6));
    }

    
    #[test]
    fn comp_list_basic() {
        let r = q("[u.name for u in $.users]");
        assert_eq!(r, json!(["Alice", "Bob", "Carol", "Dave"]));
    }

    #[test]
    fn comp_list_cond() {
        let r = q("[u.name for u in $.users if u.score > 80]");
        assert_eq!(r, json!(["Alice", "Carol"]));
    }

    #[test]
    fn comp_list_transform() {
        
        let r = q("[u.name.upper() for u in $.users if u.active == true]");
        let arr = r.as_array().unwrap();
        assert!(arr.contains(&json!("ALICE")));
        assert!(!arr.iter().any(|v| *v == json!("CAROL"))); 
    }

    #[test]
    fn comp_dict_basic() {
        let r = q("{u.id: u.name for u in $.users}");
        let obj = r.as_object().unwrap();
        assert_eq!(obj["1"], json!("Alice"));
        assert_eq!(obj["4"], json!("Dave"));
    }

    #[test]
    fn comp_dict_cond() {
        let r = q("{u.name: u.score for u in $.users if u.active}");
        let obj = r.as_object().unwrap();
        assert!(obj.contains_key("Alice"));
        assert!(!obj.contains_key("Carol")); 
    }

    #[test]
    fn comp_set_unique() {
        
        let r = q("{u.role for u in $.users}");
        let arr = r.as_array().unwrap();
        
        assert_eq!(arr.len(), 3);
    }

    #[test]
    fn comp_gen_lazy() {
        
        let r = q("(u.score for u in $.users if u.active)");
        let arr = r.as_array().unwrap();
        assert_eq!(arr.len(), 3);
    }

    
    #[test]
    fn let_simple() {
        let r = q("let admins = $.users.filter(role == \"admin\") in admins.len()");
        assert_eq!(r, json!(1));
    }

    #[test]
    fn let_nested() {
        let r = q("let active = $.users.filter(active == true) in \
             let top    = active.filter(score > 70) in \
             top.map(name)");
        let arr = r.as_array().unwrap();
        assert!(arr.contains(&json!("Alice")));
        assert!(arr.contains(&json!("Bob")));
    }

    #[test]
    fn let_complex_body() {
        let r = q("let orders = $.orders in \
             {total: orders.sum(total), \
              pending: orders.filter(status == \"pending\").len(), \
              avg: orders.avg(total)}");
        assert_eq!(r["pending"], json!(2));
    }

    
    #[test]
    fn pipe_to_method() {
        assert_eq!(q("$.users | len"), json!(4));
    }

    #[test]
    fn pipe_chain() {
        
        let r = q("$.products | filter(price < 20) | map(name) | sort");
        let arr = r.as_array().unwrap();
        assert!(arr.contains(&json!("Widget A")));
    }

    #[test]
    fn pipe_comprehension_then_method() {
        let r = q("(u.score for u in $.users if u.active) | sum");
        
        assert_eq!(r, json!(228));
    }

    
    #[test]
    fn bind_name() {
        let r = q("$.users -> users | {count: users.len(), first: users[0].name}");
        assert_eq!(r["count"], json!(4));
        assert_eq!(r["first"], json!("Alice"));
    }

    #[test]
    fn bind_object_destructure() {
        let doc = json!({"u": {"name": "Alice", "age": 30, "role": "admin"}});
        let r = vm_query("$.u -> {name, age} | f\"{name} is {age}\"", &doc).unwrap();
        assert_eq!(r, json!("Alice is 30"));
    }

    #[test]
    fn bind_object_rest() {
        let doc = json!({"obj": {"a": 1, "b": 2, "c": 3}});
        let r = vm_query("$.obj -> {a, ...rest} | rest", &doc).unwrap();
        assert_eq!(r["b"], json!(2));
        assert_eq!(r["c"], json!(3));
        assert!(r.get("a").is_none());
    }

    #[test]
    fn bind_array_destructure() {
        let doc = json!({"nums": [10, 20, 30]});
        let r = vm_query("$.nums -> [x, y, z] | x + y + z", &doc).unwrap();
        assert_eq!(r, json!(60));
    }

    
    #[test]
    fn obj_literal_computed() {
        let r = q("{
            total_users: $.users.len(),
            active_count: $.users.filter(active).len(),
            avg_score: $.users.avg(score),
            top_scorer: $.users.sort(-score).first().name
        }");
        assert_eq!(r["total_users"], json!(4));
        assert_eq!(r["active_count"], json!(3));
        assert_eq!(r["top_scorer"], json!("Alice"));
    }

    #[test]
    fn obj_optional_field() {
        
        let doc = json!({"user": {"name": "Alice"}});
        let r = vm_query("{name: $.user.name, email?: $.user.email}", &doc).unwrap();
        assert_eq!(r["name"], json!("Alice"));
        assert!(r.get("email").is_none());
    }

    #[test]
    fn obj_dynamic_key() {
        
        let doc = json!({"prefix": "user", "val": 42});
        let r = vm_query("{[$.prefix]: $.val}", &doc).unwrap();
        assert_eq!(r["user"], json!(42));
    }

    #[test]
    fn obj_spread() {
        let doc = json!({"base": {"a": 1, "b": 2}, "extra": {"c": 3}});
        let r = vm_query("{...$.base, ...$.extra, d: 4}", &doc).unwrap();
        assert_eq!(r["a"], json!(1));
        assert_eq!(r["c"], json!(3));
        assert_eq!(r["d"], json!(4));
    }

    #[test]
    fn arr_spread() {
        let doc = json!({"a": [1, 2], "b": [3, 4]});
        let r = vm_query("[0, ...$.a, ...$.b, 5]", &doc).unwrap();
        assert_eq!(r, json!([0, 1, 2, 3, 4, 5]));
    }

    
    #[test]
    fn fstring_basic() {
        let doc = json!({"u": {"name": "Alice", "score": 95}});
        let r = vm_query("f\"Hello {$.u.name}, your score is {$.u.score}\"", &doc).unwrap();
        assert_eq!(r, json!("Hello Alice, your score is 95"));
    }

    #[test]
    fn fstring_pipe_method() {
        let doc = json!({"name": "alice"});
        let r = vm_query("f\"Hello {$.name|upper}!\"", &doc).unwrap();
        assert_eq!(r, json!("Hello ALICE!"));
    }

    #[test]
    fn fstring_format_spec_float() {
        let doc = json!({"v": 3.14159});
        let r = vm_query("f\"{$.v:.2f}\"", &doc).unwrap();
        assert_eq!(r, json!("3.14"));
    }

    #[test]
    fn fstring_format_spec_padding() {
        let doc = json!({"n": 42});
        let r = vm_query("f\"{$.n:>6}\"", &doc).unwrap();
        assert_eq!(r, json!("    42"));
    }

    #[test]
    fn fstring_expression() {
        let doc = json!({"a": 3, "b": 4});
        let r = vm_query("f\"sum = {$.a + $.b}\"", &doc).unwrap();
        assert_eq!(r, json!("sum = 7"));
    }

    
    #[test]
    fn global_coalesce() {
        let doc = json!({"a": null, "b": null, "c": "found"});
        assert_eq!(
            vm_query("coalesce($.a, $.b, $.c)", &doc).unwrap(),
            json!("found")
        );
        assert_eq!(
            vm_query("coalesce($.a, \"default\")", &doc).unwrap(),
            json!("default")
        );
    }

    #[test]
    fn global_chain() {
        let doc = json!({"a": [1, 2], "b": [3, 4], "c": [5]});
        let r = vm_query("chain($.a, $.b, $.c)", &doc).unwrap();
        assert_eq!(r, json!([1, 2, 3, 4, 5]));
    }

    #[test]
    fn global_zip() {
        let doc = json!({"a": [1, 2, 3], "b": ["x", "y", "z"]});
        let r = vm_query("zip($.a, $.b)", &doc).unwrap();
        assert_eq!(r, json!([[1, "x"], [2, "y"], [3, "z"]]));
    }

    #[test]
    fn global_zip_longest() {
        let doc = json!({"a": [1, 2, 3], "b": ["x"]});
        let r = vm_query("zip_longest($.a, $.b)", &doc).unwrap();
        let arr = r.as_array().unwrap();
        assert_eq!(arr.len(), 3);
        assert_eq!(arr[1][1], json!(null));
    }

    #[test]
    fn global_product() {
        let doc = json!({"colors": ["red", "blue"], "sizes": ["S", "M"]});
        let r = vm_query("product($.colors, $.sizes)", &doc).unwrap();
        assert_eq!(r.as_array().unwrap().len(), 4);
    }

    
    #[test]
    fn method_set() {
        
        
        let doc = json!({"v": 1});
        assert_eq!(vm_query("$.v.set(42)", &doc).unwrap(), json!({"v": 42}));
        assert_eq!(vm_query("$.v | set(42)", &doc).unwrap(), json!(42));
    }

    #[test]
    fn method_update() {
        
        let doc = json!({"v": 10});
        assert_eq!(
            vm_query("$.v.update(lambda x: x * 3)", &doc).unwrap(),
            json!(30)
        );
    }

    
    #[test]
    fn arith_ops() {
        let doc = json!({"a": 10, "b": 3});
        assert_eq!(vm_query("$.a + $.b", &doc).unwrap(), json!(13));
        assert_eq!(vm_query("$.a - $.b", &doc).unwrap(), json!(7));
        assert_eq!(vm_query("$.a * $.b", &doc).unwrap(), json!(30));
        assert_eq!(
            vm_query("$.a / $.b", &doc).unwrap(),
            serde_json::to_value(10.0 / 3.0).unwrap()
        );
        assert_eq!(vm_query("$.a % $.b", &doc).unwrap(), json!(1));
    }

    #[test]
    fn string_concat() {
        let doc = json!({"a": "Hello", "b": " World"});
        assert_eq!(vm_query("$.a + $.b", &doc).unwrap(), json!("Hello World"));
    }

    #[test]
    fn unary_neg() {
        assert_eq!(q("$.users.min(score).update(lambda x: -x)"), json!(-61));
    }

    
    #[test]
    fn vm_path_cache_prefix_sharing() {
        
        
        let doc = world();
        let mut vm = VM::new();

        let r1 = vm.run_str("$.products.len()", &doc).unwrap();
        let r2 = vm.run_str("$.products.sum(price)", &doc).unwrap();
        let r3 = vm.run_str("$.products[0].meta.color", &doc).unwrap();

        assert_eq!(r1, json!(4));
        let total = r2.as_f64().unwrap();
        assert!((total - 99.47).abs() < 0.01);
        assert_eq!(r3, json!("red"));

        
        let (compile_entries, path_entries) = vm.cache_stats();
        assert_eq!(compile_entries, 3);
        
        assert!(path_entries > 0);
    }

    #[test]
    fn vm_descendant_caches_discovered_paths() {
        
        
        let doc = world();
        let mut vm = VM::new();

        
        let colors = vm.run_str("$..color", &doc).unwrap();
        assert_eq!(colors.as_array().unwrap().len(), 4);

        
        let r = vm.run_str("$.products[0].meta.color", &doc).unwrap();
        assert_eq!(r, json!("red"));

        let (_, path_entries) = vm.cache_stats();
        assert!(path_entries > 0);
    }

    #[test]
    fn vm_compile_cache_reuse() {
        let doc = world();
        let mut vm = VM::new();
        for _ in 0..5 {
            vm.run_str("$.users.filter(active).len()", &doc).unwrap();
        }
        let (compile_entries, _) = vm.cache_stats();
        assert_eq!(compile_entries, 1); 
    }

    
    #[test]
    fn complex_dashboard() {
        
        let r = q(r#"{
            active_users: $.users.filter(active).len(),
            top_users: $.users.sort(-score).first(2).map({name, score}),
            revenue: $.orders.filter(status == "delivered").sum(total),
            pending_count: $.orders.filter(status == "pending").len(),
            out_of_stock: $.products.filter(stock == 0).map(name)
        }"#);
        assert_eq!(r["active_users"], json!(3));
        assert_eq!(r["pending_count"], json!(2));
        let top = r["top_users"].as_array().unwrap();
        assert_eq!(top[0]["name"], json!("Alice"));
        assert_eq!(r["out_of_stock"], json!(["Gadget X"]));
    }

    #[test]
    fn complex_join_like() {
        
        let r = q("let users_idx = $.users.index_by(id) in \
             $.orders.map({id, total, status, \
                           user: users_idx[to_string(user_id)].name})");
        let arr = r.as_array().unwrap();
        assert_eq!(arr[0]["user"], json!("Alice"));
        assert_eq!(arr[1]["user"], json!("Bob"));
    }

    #[test]
    fn complex_pipeline_reshape() {
        
        let r = q("$.products \
             | filter(price < 30) \
             | sort(-price) \
             | first(3) \
             | map({id, name, price})");
        let arr = r.as_array().unwrap();
        assert!(arr.len() <= 3);
        
        let first_price = arr[0]["price"].as_f64().unwrap();
        assert!(first_price < 30.0);
    }

    #[test]
    fn complex_nested_comprehension() {
        
        let r = q("{o.id: o.items.len() for o in $.orders}");
        assert_eq!(r["o1"], json!(2)); 
        assert_eq!(r["o2"], json!(1));
    }

    #[test]
    fn complex_let_with_comprehension() {
        let r = q("let active_ids = [u.id for u in $.users if u.active] in \
             [o.id for o in $.orders if active_ids.includes(o.user_id)]");
        let arr = r.as_array().unwrap();
        
        assert!(arr.contains(&json!("o1")));
        assert!(arr.contains(&json!("o2")));
        assert!(!arr.contains(&json!("o4"))); 
    }
}
