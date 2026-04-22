#[cfg(test)]
mod tests {
    use serde_json::json;
    use crate::query;

    fn books() -> serde_json::Value {
        json!({
            "store": {
                "books": [
                    {"title": "Dune",        "price": 12.99, "rating": 4.8, "genre": "sci-fi",  "tags": ["sci-fi","classic"]},
                    {"title": "Foundation",  "price":  9.99, "rating": 4.5, "genre": "sci-fi",  "tags": ["sci-fi","series"]},
                    {"title": "Neuromancer", "price": 11.50, "rating": 4.2, "genre": "cyberpunk","tags": ["sci-fi","cyberpunk"]},
                    {"title": "1984",        "price":  7.99, "rating": 4.6, "genre": "dystopia", "tags": ["classic","dystopia"]},
                ]
            },
            "user": {"name": "Alice", "age": 30, "score": 85}
        })
    }

    // ── Navigation ────────────────────────────────────────────────────────────

    #[test]
    fn field_access() {
        let doc = books();
        let r = query("$.user.name", &doc).unwrap();
        assert_eq!(r, json!("Alice"));
    }

    #[test]
    fn test_playground() {
        let doc = books();
        let r = query("$..books[0].filter(title == \"1984\")[0].title", &doc).unwrap();
        assert_eq!(r, json!("1984"));
    }

    #[test]
    fn nested_field() {
        let doc = books();
        let r = query("$.store.books[0].title", &doc).unwrap();
        assert_eq!(r, json!("Dune"));
    }

    #[test]
    fn negative_index() {
        let doc = books();
        let r = query("$.store.books[-1].title", &doc).unwrap();
        assert_eq!(r, json!("1984"));
    }

    #[test]
    fn slice() {
        let doc = books();
        let r = query("$.store.books[0:2].map(title)", &doc).unwrap();
        assert_eq!(r, json!(["Dune", "Foundation"]));
    }

    #[test]
    fn descendant() {
        let doc = books();
        let r = query("$..title", &doc).unwrap();
        let titles = r.as_array().unwrap();
        assert!(titles.contains(&json!("Dune")));
        assert!(titles.contains(&json!("1984")));
    }

    #[test]
    fn optional_field_null_safe() {
        let doc = json!({"user": {"name": "Bob"}});
        let r = query("$.user?.email", &doc).unwrap();
        assert_eq!(r, json!(null));
    }

    #[test]
    fn optional_field_chain() {
        let doc = json!({"user": null});
        let r = query("$.user?.name", &doc).unwrap();
        assert_eq!(r, json!(null));
    }

    // ── Filter ────────────────────────────────────────────────────────────────

    #[test]
    fn filter_simple() {
        let doc = books();
        let r = query("$.store.books.filter(price > 10)", &doc).unwrap();
        let arr = r.as_array().unwrap();
        assert_eq!(arr.len(), 2);
    }

    #[test]
    fn filter_and() {
        let doc = books();
        let r = query("$.store.books.filter(price > 10 and rating >= 4.5)", &doc).unwrap();
        let arr = r.as_array().unwrap();
        assert_eq!(arr.len(), 1);
        assert_eq!(arr[0]["title"], json!("Dune"));
    }

    #[test]
    fn filter_lambda() {
        let doc = books();
        let r = query("$.store.books.filter(lambda b: b.price > 10)", &doc).unwrap();
        let arr = r.as_array().unwrap();
        assert_eq!(arr.len(), 2);
    }

    #[test]
    fn filter_not() {
        let doc = books();
        let r = query("$.store.books.filter(not price > 10)", &doc).unwrap();
        let arr = r.as_array().unwrap();
        assert_eq!(arr.len(), 2);
    }

    // ── Map ───────────────────────────────────────────────────────────────────

    #[test]
    fn map_pluck() {
        let doc = books();
        let r = query("$.store.books.map(title)", &doc).unwrap();
        assert_eq!(r, json!(["Dune", "Foundation", "Neuromancer", "1984"]));
    }

    #[test]
    fn map_object_shorthand() {
        let doc = books();
        let r = query("$.store.books.map({title, price})", &doc).unwrap();
        let arr = r.as_array().unwrap();
        assert_eq!(arr[0]["title"], json!("Dune"));
        assert_eq!(arr[0]["price"], json!(12.99));
    }

    #[test]
    fn map_rename() {
        let doc = books();
        let r = query("$.store.books.map({name: title, cost: price})", &doc).unwrap();
        let arr = r.as_array().unwrap();
        assert_eq!(arr[0]["name"], json!("Dune"));
        assert_eq!(arr[0]["cost"], json!(12.99));
    }

    #[test]
    fn map_computed_field() {
        let doc = books();
        let r = query("$.store.books.map({title, expensive: price > 10})", &doc).unwrap();
        let arr = r.as_array().unwrap();
        assert_eq!(arr[0]["expensive"], json!(true));
        assert_eq!(arr[1]["expensive"], json!(false));
    }

    // ── Aggregates ────────────────────────────────────────────────────────────

    #[test]
    fn len() {
        let doc = books();
        assert_eq!(query("$.store.books.len()", &doc).unwrap(), json!(4));
    }

    #[test]
    fn sum() {
        let doc = json!({"nums": [1, 2, 3, 4]});
        assert_eq!(query("$.nums.sum()", &doc).unwrap(), json!(10));
    }

    #[test]
    fn sum_field() {
        let doc = json!({"items": [{"v": 1}, {"v": 2}, {"v": 3}]});
        assert_eq!(query("$.items.sum(v)", &doc).unwrap(), json!(6));
    }

    #[test]
    fn first_last() {
        let doc = books();
        assert_eq!(query("$.store.books.first().title", &doc).unwrap(), json!("Dune"));
        assert_eq!(query("$.store.books.last().title",  &doc).unwrap(), json!("1984"));
    }

    #[test]
    fn first_n() {
        let doc = books();
        let r = query("$.store.books.first(2).map(title)", &doc).unwrap();
        assert_eq!(r, json!(["Dune", "Foundation"]));
    }

    #[test]
    fn sort_asc() {
        let doc = books();
        let r = query("$.store.books.sort(price).map(title)", &doc).unwrap();
        assert_eq!(r.as_array().unwrap()[0], json!("1984"));
    }

    #[test]
    fn sort_desc() {
        let doc = books();
        let r = query("$.store.books.sort(-price).map(title)", &doc).unwrap();
        assert_eq!(r.as_array().unwrap()[0], json!("Dune"));
    }

    // ── Null safety ───────────────────────────────────────────────────────────

    #[test]
    fn or_default() {
        let doc = json!({"user": {}});
        assert_eq!(query("$.user.name.or(\"anon\")", &doc).unwrap(), json!("anon"));
    }

    #[test]
    fn has_field() {
        let doc = json!({"user": {"name": "Alice", "email": "a@b.com"}});
        assert_eq!(query("$.user.has(\"email\")", &doc).unwrap(), json!(true));
        assert_eq!(query("$.user.has(\"phone\")", &doc).unwrap(), json!(false));
    }

    #[test]
    fn missing_field() {
        let doc = json!({"user": {"name": "Alice"}});
        assert_eq!(query("$.user.missing(\"phone\")", &doc).unwrap(), json!(true));
    }

    #[test]
    fn compact() {
        let doc = json!({"vals": [1, null, 2, null, 3]});
        assert_eq!(query("$.vals.compact()", &doc).unwrap(), json!([1, 2, 3]));
    }

    // ── Kind checks ───────────────────────────────────────────────────────────

    #[test]
    fn kind_number() {
        let doc = json!({"items": [{"v": 1}, {"v": "x"}, {"v": null}]});
        let r = query("$.items.filter(v kind number)", &doc).unwrap();
        assert_eq!(r, json!([{"v": 1}]));
    }

    #[test]
    fn kind_not_null() {
        let doc = json!({"items": [{"v": 1}, {"v": null}]});
        let r = query("$.items.filter(v kind not null)", &doc).unwrap();
        assert_eq!(r, json!([{"v": 1}]));
    }

    // ── Comprehensions ────────────────────────────────────────────────────────

    #[test]
    fn list_comp_basic() {
        let doc = books();
        let r = query("[b.title for b in $.store.books]", &doc).unwrap();
        assert_eq!(r, json!(["Dune", "Foundation", "Neuromancer", "1984"]));
    }

    #[test]
    fn list_comp_with_cond() {
        let doc = books();
        let r = query("[b.title for b in $.store.books if b.price > 10]", &doc).unwrap();
        assert_eq!(r, json!(["Dune", "Neuromancer"]));
    }

    #[test]
    fn dict_comp() {
        let doc = json!({"users": [{"id": "a1", "name": "Alice"}, {"id": "b2", "name": "Bob"}]});
        let r = query("{u.id: u.name for u in $.users}", &doc).unwrap();
        assert_eq!(r["a1"], json!("Alice"));
        assert_eq!(r["b2"], json!("Bob"));
    }

    #[test]
    fn set_comp_unique() {
        let doc = json!({"items": [{"genre": "sci-fi"}, {"genre": "sci-fi"}, {"genre": "dystopia"}]});
        let r = query("{item.genre for item in $.items}", &doc).unwrap();
        let arr = r.as_array().unwrap();
        assert_eq!(arr.len(), 2);
    }

    // ── Lambda ────────────────────────────────────────────────────────────────

    #[test]
    fn lambda_update() {
        let doc = json!({"prices": [10, 20, 30]});
        let r = query("$.prices.map(lambda p: p * 2)", &doc).unwrap();
        assert_eq!(r, json!([20, 40, 60]));
    }

    // ── Let bindings ──────────────────────────────────────────────────────────

    #[test]
    fn let_binding() {
        let doc = books();
        let r = query(
            "let expensive = $.store.books.filter(price > 10) in expensive.len()",
            &doc
        ).unwrap();
        assert_eq!(r, json!(2));
    }

    #[test]
    fn let_nested() {
        let doc = books();
        let r = query(
            "let top = $.store.books.sort(-rating).first(2) in let titles = top.map(title) in titles",
            &doc
        ).unwrap();
        let arr = r.as_array().unwrap();
        assert_eq!(arr[0], json!("Dune"));
        assert_eq!(arr[1], json!("1984"));
    }

    // ── Itertools ─────────────────────────────────────────────────────────────

    #[test]
    fn enumerate() {
        let doc = json!({"items": ["a", "b", "c"]});
        let r = query("$.items.enumerate()", &doc).unwrap();
        let arr = r.as_array().unwrap();
        assert_eq!(arr[0]["index"], json!(0));
        assert_eq!(arr[0]["value"], json!("a"));
    }

    #[test]
    fn pairwise() {
        let doc = json!({"vals": [1, 2, 3, 4]});
        let r = query("$.vals.pairwise()", &doc).unwrap();
        assert_eq!(r, json!([[1, 2], [2, 3], [3, 4]]));
    }

    #[test]
    fn window() {
        let doc = json!({"vals": [1, 2, 3, 4, 5]});
        let r = query("$.vals.window(3)", &doc).unwrap();
        assert_eq!(r, json!([[1, 2, 3], [2, 3, 4], [3, 4, 5]]));
    }

    #[test]
    fn chunk() {
        let doc = json!({"vals": [1, 2, 3, 4, 5]});
        let r = query("$.vals.chunk(2)", &doc).unwrap();
        assert_eq!(r, json!([[1, 2], [3, 4], [5]]));
    }

    #[test]
    fn accumulate() {
        let doc = json!({"vals": [1, 2, 3, 4]});
        let r = query("$.vals.accumulate(lambda acc, x: acc + x)", &doc).unwrap();
        assert_eq!(r, json!([1, 3, 6, 10]));
    }

    #[test]
    fn partition() {
        let doc = json!({"nums": [1, 2, 3, 4, 5, 6]});
        let r = query("$.nums.partition(lambda n: n % 2 == 0)", &doc).unwrap();
        assert_eq!(r["true"],  json!([2, 4, 6]));
        assert_eq!(r["false"], json!([1, 3, 5]));
    }

    #[test]
    fn takewhile() {
        let doc = json!({"vals": [1, 2, 3, 4, 5]});
        let r = query("$.vals.takewhile(lambda v: v < 4)", &doc).unwrap();
        assert_eq!(r, json!([1, 2, 3]));
    }

    #[test]
    fn dropwhile() {
        let doc = json!({"vals": [1, 2, 3, 4, 5]});
        let r = query("$.vals.dropwhile(lambda v: v < 3)", &doc).unwrap();
        assert_eq!(r, json!([3, 4, 5]));
    }

    #[test]
    fn fused_filter_drop_while() {
        let doc = json!({"vals": [1, 2, 3, 4, 5, 6]});
        let r = query("$.vals.filter(lambda v: v > 1).dropwhile(lambda v: v < 4)", &doc).unwrap();
        assert_eq!(r, json!([4, 5, 6]));
    }

    #[test]
    fn fused_map_unique() {
        let doc = json!({"xs": [1, 2, 2, 3, 3, 3]});
        let r = query("$.xs.map(lambda v: v * 2).unique()", &doc).unwrap();
        assert_eq!(r, json!([2, 4, 6]));
    }

    // ── Global functions ──────────────────────────────────────────────────────

    #[test]
    fn coalesce() {
        let doc = json!({"a": null, "b": null, "c": 42});
        assert_eq!(query("coalesce($.a, $.b, $.c)", &doc).unwrap(), json!(42));
        assert_eq!(query("coalesce($.a, $.b, 99)", &doc).unwrap(), json!(99));
    }

    #[test]
    fn chain_arrays() {
        let doc = json!({"a": [1, 2], "b": [3, 4]});
        let r = query("chain($.a, $.b)", &doc).unwrap();
        assert_eq!(r, json!([1, 2, 3, 4]));
    }

    #[test]
    fn zip_arrays() {
        let doc = json!({"a": [1, 2, 3], "b": ["x", "y", "z"]});
        let r = query("zip($.a, $.b)", &doc).unwrap();
        assert_eq!(r, json!([[1, "x"], [2, "y"], [3, "z"]]));
    }

    #[test]
    fn product() {
        let doc = json!({"colors": ["red", "blue"], "sizes": ["S", "M"]});
        let r = query("product($.colors, $.sizes)", &doc).unwrap();
        let arr = r.as_array().unwrap();
        assert_eq!(arr.len(), 4);
    }

    // ── Object construction ───────────────────────────────────────────────────

    #[test]
    fn object_construction() {
        let doc = books();
        let r = query("{total: $.store.books.sum(price), count: $.store.books.len()}", &doc).unwrap();
        assert_eq!(r["count"], json!(4));
    }

    #[test]
    fn nested_obj_construct_indefinite() {
        let doc = json!({
            "books": [{"t":"x"},{"t":"y"}],
            "another": {"field": 42},
            "deep": {"a": {"b": {"c": "leaf"}}}
        });
        let r = query("{a: $.books, b: {c: $.another.field}}", &doc).unwrap();
        assert_eq!(r, json!({"a":[{"t":"x"},{"t":"y"}],"b":{"c":42}}));
        let r2 = query("{x: {y: {z: $.deep.a.b.c, arr: [1, $.another.field, {w: $.books[0].t}]}}}", &doc).unwrap();
        assert_eq!(r2, json!({"x":{"y":{"z":"leaf","arr":[1,42,{"w":"x"}]}}}));
    }

    #[test]
    fn optional_field_omitted() {
        let doc = json!({"user": {"name": "Alice"}});
        let _r = query("$.user.map({name, email?})", &doc);
        // map on non-array returns error, but test optional field in object directly
        let r2 = query("{name: $.user.name, email?: $.user.email}", &doc).unwrap();
        assert!(r2.get("email").is_none());
        assert_eq!(r2["name"], json!("Alice"));
    }

    // ── Pipe operator ─────────────────────────────────────────────────────────

    #[test]
    fn pipe_to_method() {
        let doc = books();
        let r = query("$.store.books | len", &doc).unwrap();
        assert_eq!(r, json!(4));
    }

    #[test]
    fn gen_comp_pipe() {
        let doc = books();
        let r = query("(b.price for b in $.store.books if b.price > 10) | len", &doc).unwrap();
        assert_eq!(r, json!(2));
    }

    // ── Null-coalesce (?|) ────────────────────────────────────────────────────

    #[test]
    fn null_coalesce_basic() {
        let doc = json!({"a": null, "b": 42});
        assert_eq!(query("$.a ?| $.b", &doc).unwrap(), json!(42));
    }

    #[test]
    fn null_coalesce_non_null_short_circuits() {
        let doc = json!({"a": 1, "b": 99});
        assert_eq!(query("$.a ?| $.b", &doc).unwrap(), json!(1));
    }

    #[test]
    fn null_coalesce_chain() {
        let doc = json!({"a": null, "b": null, "c": "found"});
        assert_eq!(query("$.a ?| $.b ?| $.c", &doc).unwrap(), json!("found"));
    }

    // ── Bind operator (->) ────────────────────────────────────────────────────

    #[test]
    fn bind_simple_name() {
        let doc = books();
        // bind labels current value as `books`, then use it twice
        let r = query("$.store.books -> books | {count: books.len(), first: books[0].title}", &doc).unwrap();
        assert_eq!(r["count"], json!(4));
        assert_eq!(r["first"], json!("Dune"));
    }

    #[test]
    fn bind_object_destructure() {
        let doc = json!({"user": {"name": "Alice", "age": 30, "role": "admin"}});
        let r = query("$.user -> {name, age} | {greeting: name, years: age}", &doc).unwrap();
        assert_eq!(r["greeting"], json!("Alice"));
        assert_eq!(r["years"], json!(30));
    }

    #[test]
    fn bind_object_rest() {
        let doc = json!({"obj": {"a": 1, "b": 2, "c": 3}});
        let r = query("$.obj -> {a, ...rest} | rest.len()", &doc).unwrap();
        assert_eq!(r, json!(2));
    }

    #[test]
    fn bind_array_destructure() {
        let doc = json!({"nums": [10, 20, 30]});
        let r = query("$.nums -> [x, y, z] | x + y + z", &doc).unwrap();
        assert_eq!(r, json!(60));
    }

    // ── Object spread ─────────────────────────────────────────────────────────

    #[test]
    fn object_spread() {
        let doc = json!({"base": {"a": 1, "b": 2}, "extra": {"c": 3}});
        let r = query("{...$.base, ...$.extra}", &doc).unwrap();
        assert_eq!(r["a"], json!(1));
        assert_eq!(r["b"], json!(2));
        assert_eq!(r["c"], json!(3));
    }

    #[test]
    fn object_spread_override() {
        let doc = json!({"base": {"a": 1, "b": 2}});
        let r = query("{...$.base, b: 99}", &doc).unwrap();
        assert_eq!(r["a"], json!(1));
        assert_eq!(r["b"], json!(99));
    }

    // ── Array spread ──────────────────────────────────────────────────────────

    #[test]
    fn array_spread() {
        let doc = json!({"a": [1, 2], "b": [3, 4]});
        let r = query("[...$.a, ...$.b]", &doc).unwrap();
        assert_eq!(r, json!([1, 2, 3, 4]));
    }

    #[test]
    fn array_spread_with_literal() {
        let doc = json!({"items": [2, 3]});
        let r = query("[1, ...$.items, 4]", &doc).unwrap();
        assert_eq!(r, json!([1, 2, 3, 4]));
    }

    // ── F-strings ─────────────────────────────────────────────────────────────

    #[test]
    fn fstring_basic() {
        let doc = json!({"user": {"name": "Alice", "age": 30}});
        let r = query("f\"Hello {$.user.name}!\"", &doc).unwrap();
        assert_eq!(r, json!("Hello Alice!"));
    }

    #[test]
    fn fstring_multiple_interp() {
        let doc = json!({"user": {"name": "Bob", "score": 95}});
        let r = query("f\"{$.user.name} scored {$.user.score}\"", &doc).unwrap();
        assert_eq!(r, json!("Bob scored 95"));
    }

    #[test]
    fn fstring_pipe_method() {
        let doc = json!({"name": "alice"});
        let r = query("f\"Hello {$.name|upper}!\"", &doc).unwrap();
        assert_eq!(r, json!("Hello ALICE!"));
    }

    // ── String methods ────────────────────────────────────────────────────────

    #[test]
    fn str_upper_lower() {
        let doc = json!({"s": "Hello World"});
        assert_eq!(query("$.s.upper()", &doc).unwrap(), json!("HELLO WORLD"));
        assert_eq!(query("$.s.lower()", &doc).unwrap(), json!("hello world"));
    }

    #[test]
    fn str_trim() {
        let doc = json!({"s": "  hello  "});
        assert_eq!(query("$.s.trim()", &doc).unwrap(), json!("hello"));
        assert_eq!(query("$.s.trim_left()", &doc).unwrap(), json!("hello  "));
        assert_eq!(query("$.s.trim_right()", &doc).unwrap(), json!("  hello"));
    }

    #[test]
    fn str_pad() {
        let doc = json!({"s": "hi"});
        assert_eq!(query("$.s.pad_left(5)", &doc).unwrap(), json!("   hi"));
        assert_eq!(query("$.s.pad_right(5)", &doc).unwrap(), json!("hi   "));
        assert_eq!(query("$.s.pad_left(5, \"0\")", &doc).unwrap(), json!("000hi"));
    }

    #[test]
    fn str_starts_ends_with() {
        let doc = json!({"s": "hello world"});
        assert_eq!(query("$.s.starts_with(\"hello\")", &doc).unwrap(), json!(true));
        assert_eq!(query("$.s.ends_with(\"world\")", &doc).unwrap(), json!(true));
        assert_eq!(query("$.s.starts_with(\"world\")", &doc).unwrap(), json!(false));
    }

    #[test]
    fn str_replace() {
        let doc = json!({"s": "foo foo foo"});
        // 2-arg string replace — untouched by the v2 chain-write classifier
        assert_eq!(query("$.s.replace(\"foo\", \"bar\")", &doc).unwrap(), json!("bar foo foo"));
        assert_eq!(query("$.s.replace_all(\"foo\", \"bar\")", &doc).unwrap(), json!("bar bar bar"));
    }

    #[test]
    fn str_split() {
        let doc = json!({"s": "a,b,c"});
        assert_eq!(query("$.s.split(\",\")", &doc).unwrap(), json!(["a", "b", "c"]));
    }

    #[test]
    fn str_index_of() {
        let doc = json!({"s": "hello world"});
        assert_eq!(query("$.s.index_of(\"world\")", &doc).unwrap(), json!(6));
        assert_eq!(query("$.s.index_of(\"xyz\")", &doc).unwrap(), json!(-1));
    }

    #[test]
    fn str_slice() {
        let doc = json!({"s": "hello"});
        assert_eq!(query("$.s.slice(1, 4)", &doc).unwrap(), json!("ell"));
        assert_eq!(query("$.s.slice(2)", &doc).unwrap(), json!("llo"));
    }

    #[test]
    fn str_repeat() {
        let doc = json!({"s": "ab"});
        assert_eq!(query("$.s.repeat(3)", &doc).unwrap(), json!("ababab"));
    }

    #[test]
    fn str_strip_prefix_suffix() {
        let doc = json!({"s": "foobar"});
        assert_eq!(query("$.s.strip_prefix(\"foo\")", &doc).unwrap(), json!("bar"));
        assert_eq!(query("$.s.strip_suffix(\"bar\")", &doc).unwrap(), json!("foo"));
    }

    #[test]
    fn str_to_number() {
        let doc = json!({"s": "42"});
        assert_eq!(query("$.s.to_number()", &doc).unwrap(), json!(42));
    }

    #[test]
    fn str_base64_roundtrip() {
        let doc = json!({"s": "hello world"});
        let encoded = query("$.s.to_base64()", &doc).unwrap();
        let decoded_doc = json!({"s": encoded});
        let r = query("$.s.from_base64()", &decoded_doc).unwrap();
        assert_eq!(r, json!("hello world"));
    }

    #[test]
    fn str_url_encode_decode() {
        let doc = json!({"s": "hello world&foo=bar"});
        let encoded = query("$.s.url_encode()", &doc).unwrap();
        assert!(encoded.as_str().unwrap().contains("%20") || encoded.as_str().unwrap().contains("+"));
        let encoded_doc = json!({"s": encoded});
        let decoded = query("$.s.url_decode()", &encoded_doc).unwrap();
        assert_eq!(decoded, json!("hello world&foo=bar"));
    }

    #[test]
    fn str_html_escape() {
        let doc = json!({"s": "<b>Hello & World</b>"});
        let r = query("$.s.html_escape()", &doc).unwrap();
        assert_eq!(r, json!("&lt;b&gt;Hello &amp; World&lt;/b&gt;"));
    }

    #[test]
    fn str_lines_words_chars() {
        let doc = json!({"s": "a b\nc d"});
        let lines = query("$.s.lines()", &doc).unwrap();
        assert_eq!(lines, json!(["a b", "c d"]));
        let words = query("$.s.words()", &doc).unwrap();
        assert_eq!(words, json!(["a", "b", "c", "d"]));
    }

    #[test]
    fn str_capitalize() {
        let doc = json!({"s": "hello world"});
        assert_eq!(query("$.s.capitalize()", &doc).unwrap(), json!("Hello world"));
    }

    #[test]
    fn str_title_case() {
        let doc = json!({"s": "hello world"});
        assert_eq!(query("$.s.title_case()", &doc).unwrap(), json!("Hello World"));
    }

    // ── JSON field manipulation ───────────────────────────────────────────────

    #[test]
    fn pick_fields() {
        let doc = json!({"user": {"name": "Alice", "age": 30, "password": "secret"}});
        let r = query("$.user.pick(\"name\", \"age\")", &doc).unwrap();
        assert_eq!(r["name"], json!("Alice"));
        assert_eq!(r["age"], json!(30));
        assert!(r.get("password").is_none());
    }

    #[test]
    fn omit_fields() {
        let doc = json!({"user": {"name": "Alice", "password": "secret"}});
        let r = query("$.user.omit(\"password\")", &doc).unwrap();
        assert_eq!(r["name"], json!("Alice"));
        assert!(r.get("password").is_none());
    }

    #[test]
    fn rename_fields() {
        let doc = json!({"obj": {"old_name": "value"}});
        let r = query("$.obj.rename({old_name: \"new_name\"})", &doc).unwrap();
        assert_eq!(r["new_name"], json!("value"));
        assert!(r.get("old_name").is_none());
    }

    #[test]
    fn merge_objects() {
        // chain form is a write terminal now — pipe form preserves old semantics
        let doc = json!({"a": {"x": 1}, "b": {"y": 2}});
        let r = query("$.a | merge($.b)", &doc).unwrap();
        assert_eq!(r["x"], json!(1));
        assert_eq!(r["y"], json!(2));
    }

    #[test]
    fn deep_merge_objects() {
        let doc = json!({"a": {"x": {"p": 1}}, "b": {"x": {"q": 2}, "y": 3}});
        let r = query("$.a | deep_merge($.b)", &doc).unwrap();
        assert_eq!(r["x"]["p"], json!(1));
        assert_eq!(r["x"]["q"], json!(2));
        assert_eq!(r["y"], json!(3));
    }

    #[test]
    fn defaults_fill_nulls() {
        let doc = json!({"obj": {"a": 1, "b": null}, "defs": {"b": 99, "c": 100}});
        let r = query("$.obj.defaults($.defs)", &doc).unwrap();
        assert_eq!(r["a"], json!(1));
        assert_eq!(r["b"], json!(99));
        assert_eq!(r["c"], json!(100));
    }

    #[test]
    fn transform_keys() {
        let doc = json!({"obj": {"foo_bar": 1, "baz_qux": 2}});
        let r = query("$.obj.transform_keys(lambda k: k.upper())", &doc).unwrap();
        assert!(r.get("FOO_BAR").is_some() || r.get("BAZ_QUX").is_some());
    }

    #[test]
    fn transform_values() {
        let doc = json!({"obj": {"a": 1, "b": 2, "c": 3}});
        let r = query("$.obj.transform_values(lambda v: v * 10)", &doc).unwrap();
        assert_eq!(r["a"], json!(10));
        assert_eq!(r["b"], json!(20));
    }

    #[test]
    fn filter_keys_test() {
        let doc = json!({"obj": {"name": "Alice", "_private": "x", "_secret": "y"}});
        let r = query("$.obj.filter_keys(lambda k: not k.starts_with(\"_\"))", &doc).unwrap();
        assert!(r.get("name").is_some());
        assert!(r.get("_private").is_none());
    }

    #[test]
    fn filter_values_test() {
        let doc = json!({"obj": {"a": 1, "b": null, "c": 3}});
        let r = query("$.obj.filter_values(lambda v: v kind not null)", &doc).unwrap();
        assert!(r.get("a").is_some());
        assert!(r.get("b").is_none());
        assert!(r.get("c").is_some());
    }

    #[test]
    fn invert_object() {
        let doc = json!({"obj": {"a": "x", "b": "y"}});
        let r = query("$.obj.invert()", &doc).unwrap();
        assert_eq!(r["x"], json!("a"));
        assert_eq!(r["y"], json!("b"));
    }

    #[test]
    fn to_pairs_from_pairs() {
        let doc = json!({"obj": {"a": 1, "b": 2}});
        let pairs = query("$.obj.to_pairs()", &doc).unwrap();
        let arr = pairs.as_array().unwrap();
        assert_eq!(arr.len(), 2);
        // round-trip
        let pairs_doc = json!({"pairs": pairs});
        let restored = query("$.pairs.from_pairs()", &pairs_doc).unwrap();
        assert_eq!(restored["a"], json!(1));
        assert_eq!(restored["b"], json!(2));
    }

    // ── Path operations ───────────────────────────────────────────────────────

    #[test]
    fn get_path_op() {
        let doc = json!({"a": {"b": {"c": 42}}});
        assert_eq!(query("$.get_path(\"a.b.c\")", &doc).unwrap(), json!(42));
    }

    #[test]
    fn set_path_op() {
        let doc = json!({"a": {"b": 1}});
        let r = query("$.set_path(\"a.b\", 99)", &doc).unwrap();
        assert_eq!(r["a"]["b"], json!(99));
    }

    #[test]
    fn del_path_op() {
        let doc = json!({"a": {"b": 1, "c": 2}});
        let r = query("$.del_path(\"a.b\")", &doc).unwrap();
        assert!(r["a"].get("b").is_none());
        assert_eq!(r["a"]["c"], json!(2));
    }

    #[test]
    fn has_path_op() {
        let doc = json!({"a": {"b": {"c": 1}}});
        assert_eq!(query("$.has_path(\"a.b.c\")", &doc).unwrap(), json!(true));
        assert_eq!(query("$.has_path(\"a.x.y\")", &doc).unwrap(), json!(false));
    }

    #[test]
    fn flatten_keys_op() {
        let doc = json!({"a": {"b": {"c": 1}, "d": 2}});
        let r = query("$.flatten_keys()", &doc).unwrap();
        assert_eq!(r["a.b.c"], json!(1));
        assert_eq!(r["a.d"], json!(2));
    }

    #[test]
    fn unflatten_keys_op() {
        let doc = json!({"flat": {"a.b.c": 1, "a.d": 2}});
        let r = query("$.flat.unflatten_keys()", &doc).unwrap();
        assert_eq!(r["a"]["b"]["c"], json!(1));
        assert_eq!(r["a"]["d"], json!(2));
    }

    // ── Set operations ────────────────────────────────────────────────────────

    #[test]
    fn set_diff() {
        let doc = json!({"a": [1, 2, 3, 4], "b": [2, 4]});
        let r = query("$.a.diff($.b)", &doc).unwrap();
        assert_eq!(r, json!([1, 3]));
    }

    #[test]
    fn set_intersect() {
        let doc = json!({"a": [1, 2, 3], "b": [2, 3, 4]});
        let r = query("$.a.intersect($.b)", &doc).unwrap();
        assert_eq!(r, json!([2, 3]));
    }

    #[test]
    fn set_union() {
        let doc = json!({"a": [1, 2, 3], "b": [3, 4, 5]});
        let r = query("$.a.union($.b)", &doc).unwrap();
        let arr = r.as_array().unwrap();
        assert_eq!(arr.len(), 5);
        assert!(arr.contains(&json!(1)));
        assert!(arr.contains(&json!(5)));
    }

    // ── Type / conversion ─────────────────────────────────────────────────────

    #[test]
    fn type_method() {
        let doc = json!({"n": 42, "s": "hello", "a": [1], "o": {}, "b": true, "z": null});
        assert_eq!(query("$.n.type()", &doc).unwrap(), json!("number"));
        assert_eq!(query("$.s.type()", &doc).unwrap(), json!("string"));
        assert_eq!(query("$.a.type()", &doc).unwrap(), json!("array"));
        assert_eq!(query("$.o.type()", &doc).unwrap(), json!("object"));
        assert_eq!(query("$.b.type()", &doc).unwrap(), json!("bool"));
        assert_eq!(query("$.z.type()", &doc).unwrap(), json!("null"));
    }

    #[test]
    fn from_json_to_json() {
        let doc = json!({"s": "{\"x\":1}"});
        let parsed = query("$.s.from_json()", &doc).unwrap();
        assert_eq!(parsed["x"], json!(1));
        let serialized = query("$.s.from_json().to_json()", &doc).unwrap();
        assert!(serialized.as_str().unwrap().contains("\"x\""));
    }

    // ── New syntax: DescendAll, InlineFilter, Quantifier ──────────────────────

    #[test]
    fn inline_filter_basic() {
        let doc = books();
        // $.store.books{price > 10} — filter books array inline
        let r = query("$.store.books{price > 10}", &doc).unwrap();
        let arr = r.as_array().unwrap();
        assert_eq!(arr.len(), 2);
        assert!(arr.iter().all(|b| b["price"].as_f64().unwrap() > 10.0));
    }

    #[test]
    fn quantifier_first() {
        let doc = books();
        // first book over $10
        let r = query("$.store.books{price > 10}?", &doc).unwrap();
        assert_eq!(r["title"], json!("Dune"));
    }

    #[test]
    fn quantifier_one_ok() {
        let doc = books();
        let r = query("$.store.books{title == \"1984\"}!", &doc).unwrap();
        assert_eq!(r["title"], json!("1984"));
    }

    #[test]
    fn quantifier_one_error() {
        let doc = books();
        // more than one match → error
        let r = query("$.store.books{price > 10}!", &doc);
        assert!(r.is_err());
    }

    #[test]
    fn descend_all_inline_filter() {
        let doc = books();
        // $..{title == "1984"}!.title — the clean recursive-descent style
        let r = query("$.store..{title == \"1984\"}!.title", &doc).unwrap();
        assert_eq!(r, json!("1984"));
    }

    #[test]
    fn descend_all_collect() {
        let doc = json!({"a": {"b": 1, "c": 2}, "d": 3});
        // $.. collects all values
        let r = query("$..", &doc).unwrap();
        let arr = r.as_array().unwrap();
        // should include objects and scalars
        assert!(arr.len() > 1);
    }

    // ── Peephole fusion passes ────────────────────────────────────────────────

    #[test]
    fn fusion_drop_noop_before_len() {
        use crate::vm::{Compiler, Opcode, BuiltinMethod};
        // sort → len: sort is dropped (sort preserves length)
        let p1 = Compiler::compile_str("$.xs.sort().len()").unwrap();
        let sort_ct = p1.ops.iter().filter(|o|
            matches!(o, Opcode::CallMethod(c) if c.method == BuiltinMethod::Sort)).count();
        assert_eq!(sort_ct, 0, "sort should be dropped before length; ops: {:?}", p1.ops);

        // map → count: map is dropped
        let p2 = Compiler::compile_str("$.xs.map(@ * 2).count()").unwrap();
        let map_ct = p2.ops.iter().filter(|o|
            matches!(o, Opcode::CallMethod(c) if c.method == BuiltinMethod::Map)).count();
        assert_eq!(map_ct, 0, "map should be dropped before count; ops: {:?}", p2.ops);
    }

    #[test]
    fn fusion_drop_noop_before_len_semantics() {
        let doc = json!({"xs": [3, 1, 4, 1, 5, 9, 2, 6]});
        assert_eq!(query("$.xs.sort().len()", &doc).unwrap(), json!(8));
        assert_eq!(query("$.xs.reverse().count()", &doc).unwrap(), json!(8));
        assert_eq!(query("$.xs.map(@ * 2).len()", &doc).unwrap(), json!(8));
    }

    #[test]
    fn fusion_map_filter_opcode_emitted() {
        use crate::vm::{Compiler, Opcode};
        let prog = Compiler::compile_str("$.xs.map(@ * 2).filter(@ > 5)").unwrap();
        let has_mf = prog.ops.iter().any(|o| matches!(o, Opcode::MapFilter { .. }));
        assert!(has_mf, "MapFilter not emitted; ops: {:?}", prog.ops);
    }

    #[test]
    fn fusion_map_filter_semantics() {
        let doc = json!({"xs": [1, 2, 3, 4, 5]});
        let r = query("$.xs.map(@ * 2).filter(@ > 5)", &doc).unwrap();
        assert_eq!(r, json!([6, 8, 10]));
    }

    #[test]
    fn fusion_field_chain_opcode_emitted() {
        use crate::vm::{Compiler, Opcode};
        // `.first()` returns object; `.a.b.c` mid-program can't become RootChain
        // because of the intervening method call.  Expect FieldChain instead.
        let prog = Compiler::compile_str("$.items.first().a.b.c").unwrap();
        let has_fc = prog.ops.iter().any(|o| matches!(o, Opcode::FieldChain(c) if c.len() == 3));
        let get_field_count = prog.ops.iter().filter(|o| matches!(o, Opcode::GetField(_))).count();
        assert!(has_fc, "FieldChain not emitted; ops: {:?}", prog.ops);
        assert_eq!(get_field_count, 0, "residual GetField after fusion: {:?}", prog.ops);
    }

    #[test]
    fn fusion_opt_field_absorbed_into_field_chain() {
        use crate::vm::{Compiler, Opcode};
        // Mid-program OptField chain (receiver from method call so nullness
        // analyzer can't prove the shape) should still collapse into one
        // FieldChain — null propagates through `get_field` correctly.
        let prog = Compiler::compile_str("$.items.first()?.a?.b?.c").unwrap();
        let has_fc = prog.ops.iter().any(|o| matches!(o, Opcode::FieldChain(c) if c.len() >= 2));
        let residual = prog.ops.iter().filter(|o|
            matches!(o, Opcode::OptField(_) | Opcode::GetField(_))).count();
        assert!(has_fc, "FieldChain not emitted; ops: {:?}", prog.ops);
        assert_eq!(residual, 0, "residual per-step field ops: {:?}", prog.ops);
    }

    #[test]
    fn fusion_opt_field_chain_semantics() {
        let doc = json!({"items": [{"a": {"b": {"c": 42}}}]});
        let r = query("$.items.first()?.a?.b?.c", &doc).unwrap();
        assert_eq!(r, json!(42));
        let missing = json!({"items": [{"a": null}]});
        let r2 = query("$.items.first()?.a?.b?.c", &missing).unwrap();
        assert!(r2.is_null());
    }

    #[test]
    fn fusion_field_chain_semantics() {
        let doc = json!({"items": [{"a": {"b": {"c": 42}}}]});
        let r = query("$.items.first().a.b.c", &doc).unwrap();
        assert_eq!(r, json!(42));
    }

    #[test]
    fn fusion_find_first_opcode_emitted() {
        use crate::vm::{Compiler, Opcode};
        let prog = Compiler::compile_str("$.books{price > 10}?").unwrap();
        let has_find_first = prog.ops.iter().any(|o| matches!(o, Opcode::FindFirst(_)));
        assert!(has_find_first, "FindFirst opcode not emitted; got: {:?}", prog.ops);
    }

    #[test]
    fn fusion_find_one_opcode_emitted() {
        use crate::vm::{Compiler, Opcode};
        let prog = Compiler::compile_str("$.books{title == \"x\"}!").unwrap();
        let has_find_one = prog.ops.iter().any(|o| matches!(o, Opcode::FindOne(_)));
        assert!(has_find_one, "FindOne opcode not emitted; got: {:?}", prog.ops);
    }

    #[test]
    fn fusion_find_first_from_filter_method() {
        use crate::vm::{Compiler, Opcode};
        let prog = Compiler::compile_str("$.books.filter(price > 10)?").unwrap();
        let has_find_first = prog.ops.iter().any(|o| matches!(o, Opcode::FindFirst(_)));
        assert!(has_find_first, "FindFirst should fuse from .filter() too");
    }

    #[test]
    fn redundant_reverse_eliminated() {
        use crate::vm::{Compiler, Opcode, BuiltinMethod};
        let prog = Compiler::compile_str("$.books.reverse().reverse()").unwrap();
        let reverse_count = prog.ops.iter().filter(|o| matches!(o,
            Opcode::CallMethod(c) if c.method == BuiltinMethod::Reverse
        )).count();
        assert_eq!(reverse_count, 0, "adjacent reverse+reverse should cancel");
    }

    #[test]
    fn redundant_unique_collapsed() {
        use crate::vm::{Compiler, Opcode, BuiltinMethod};
        let prog = Compiler::compile_str("$.books.unique().unique()").unwrap();
        let unique_count = prog.ops.iter().filter(|o| matches!(o,
            Opcode::CallMethod(c) if c.method == BuiltinMethod::Unique
        )).count();
        assert_eq!(unique_count, 1, "duplicate unique() should collapse");
    }

    #[test]
    fn bool_short_circuit_folded() {
        use crate::vm::{Compiler, Opcode};
        let prog = Compiler::compile_str("false and $.expensive.deeply.nested").unwrap();
        // Should be one PushBool(false) with no AndOp or field access
        let has_and = prog.ops.iter().any(|o| matches!(o, Opcode::AndOp(_)));
        assert!(!has_and, "false and _ should fold to just PushBool(false)");
    }

    // ── Semantic equivalence: fused vs unfused should produce same result ──

    #[test]
    fn find_first_matches_semantics() {
        let doc = books();
        // inline filter + ? matches first result of unfused filter
        let fused = query("$.store.books{price > 10}?", &doc).unwrap();
        assert_eq!(fused["title"], json!("Dune"));
    }

    #[test]
    fn find_one_error_on_multiple() {
        let doc = books();
        let r = query("$.store.books{price > 10}!", &doc);
        assert!(r.is_err());
    }

    #[test]
    fn find_one_error_on_zero() {
        let doc = books();
        let r = query("$.store.books{price > 9999}!", &doc);
        assert!(r.is_err());
    }

    // ── Analysis module + kind-check folding ──────────────────────────────────

    #[test]
    fn kind_check_literal_fold_int() {
        use crate::vm::{Compiler, Opcode};
        let prog = Compiler::compile_str("42 kind number").unwrap();
        let has_kind = prog.ops.iter().any(|o| matches!(o, Opcode::KindCheck { .. }));
        assert!(!has_kind, "literal kind check should fold away");
        assert!(matches!(prog.ops.first(), Some(Opcode::PushBool(true))));
    }

    #[test]
    fn kind_check_literal_fold_mismatch() {
        use crate::vm::{Compiler, Opcode};
        let prog = Compiler::compile_str("\"hi\" kind number").unwrap();
        assert!(matches!(prog.ops.first(), Some(Opcode::PushBool(false))));
        assert!(!prog.ops.iter().any(|o| matches!(o, Opcode::KindCheck { .. })));
    }

    #[test]
    fn kind_check_literal_negate() {
        use crate::vm::{Compiler, Opcode};
        let prog = Compiler::compile_str("42 kind not string").unwrap();
        assert!(matches!(prog.ops.first(), Some(Opcode::PushBool(true))));
        assert!(!prog.ops.iter().any(|o| matches!(o, Opcode::KindCheck { .. })));
    }

    #[test]
    fn analysis_infer_result_type() {
        use crate::analysis::{infer_result_type, VType};
        use crate::vm::Compiler;
        let p = Compiler::compile_str("42 + 1").unwrap();
        let av = infer_result_type(&p);
        assert_eq!(av.ty, VType::Int);
        let p = Compiler::compile_str("[1,2,3]").unwrap();
        assert_eq!(infer_result_type(&p).ty, VType::Arr);
        let p = Compiler::compile_str("\"hi\"").unwrap();
        assert_eq!(infer_result_type(&p).ty, VType::Str);
    }

    #[test]
    fn analysis_count_ident_uses() {
        use crate::analysis::count_ident_uses;
        use crate::vm::Compiler;
        let p = Compiler::compile_str("let x = 10 in x + x + 1").unwrap();
        assert_eq!(count_ident_uses(&p, "x"), 2);
        let p = Compiler::compile_str("let y = 10 in 42").unwrap();
        assert_eq!(count_ident_uses(&p, "y"), 0);
    }

    #[test]
    fn analysis_collect_accessed_fields() {
        use crate::analysis::collect_accessed_fields;
        use crate::vm::Compiler;
        let p = Compiler::compile_str("$.store.books.map(@.title)").unwrap();
        let fields = collect_accessed_fields(&p);
        assert!(fields.iter().any(|f| f.as_ref() == "store"));
        assert!(fields.iter().any(|f| f.as_ref() == "books"));
        assert!(fields.iter().any(|f| f.as_ref() == "title"));
    }

    #[test]
    fn analysis_program_signature_stable() {
        use crate::analysis::program_signature;
        use crate::vm::Compiler;
        let a = Compiler::compile_str("$.x.y + 1").unwrap();
        let b = Compiler::compile_str("$.x.y + 1").unwrap();
        assert_eq!(program_signature(&a), program_signature(&b));
        let c = Compiler::compile_str("$.x.y + 2").unwrap();
        assert_ne!(program_signature(&a), program_signature(&c));
    }

    #[test]
    fn dead_let_eliminated() {
        use crate::vm::{Compiler, Opcode};
        let prog = Compiler::compile_str("let x = 100 in 42").unwrap();
        let has_let = prog.ops.iter().any(|o| matches!(o, Opcode::LetExpr { .. }));
        assert!(!has_let, "unused let should be eliminated");
        assert!(matches!(prog.ops.as_ref(), [Opcode::PushInt(42)]));
    }

    #[test]
    fn dead_let_kept_when_used() {
        use crate::vm::{Compiler, Opcode};
        let prog = Compiler::compile_str("let x = 10 in x + 1").unwrap();
        assert!(prog.ops.iter().any(|o| matches!(o, Opcode::LetExpr { .. })));
    }

    #[test]
    fn const_fold_comparison() {
        use crate::vm::{Compiler, Opcode};
        let prog = Compiler::compile_str("1 < 2").unwrap();
        assert!(matches!(prog.ops.as_ref(), [Opcode::PushBool(true)]));
        let prog = Compiler::compile_str("5 == 5").unwrap();
        assert!(matches!(prog.ops.as_ref(), [Opcode::PushBool(true)]));
        let prog = Compiler::compile_str("\"a\" == \"b\"").unwrap();
        assert!(matches!(prog.ops.as_ref(), [Opcode::PushBool(false)]));
    }

    #[test]
    fn const_fold_mixed_int_float_arith() {
        use crate::vm::{Compiler, Opcode};
        let prog = Compiler::compile_str("1 + 2.5").unwrap();
        assert!(matches!(prog.ops.as_ref(), [Opcode::PushFloat(f)] if (*f - 3.5).abs() < 1e-9),
                "1 + 2.5 should fold to 3.5; got {:?}", prog.ops);
        let prog = Compiler::compile_str("2.0 * 3").unwrap();
        assert!(matches!(prog.ops.as_ref(), [Opcode::PushFloat(f)] if (*f - 6.0).abs() < 1e-9));
        let prog = Compiler::compile_str("10 / 4.0").unwrap();
        assert!(matches!(prog.ops.as_ref(), [Opcode::PushFloat(f)] if (*f - 2.5).abs() < 1e-9));
    }

    #[test]
    fn const_fold_mixed_int_float_cmp() {
        use crate::vm::{Compiler, Opcode};
        let prog = Compiler::compile_str("1 < 2.5").unwrap();
        assert!(matches!(prog.ops.as_ref(), [Opcode::PushBool(true)]));
        let prog = Compiler::compile_str("3.14 > 3").unwrap();
        assert!(matches!(prog.ops.as_ref(), [Opcode::PushBool(true)]));
        let prog = Compiler::compile_str("2.0 <= 2").unwrap();
        assert!(matches!(prog.ops.as_ref(), [Opcode::PushBool(true)]));
    }

    #[test]
    fn const_fold_unary() {
        use crate::vm::{Compiler, Opcode};
        let prog = Compiler::compile_str("not true").unwrap();
        assert!(matches!(prog.ops.as_ref(), [Opcode::PushBool(false)]));
        let prog = Compiler::compile_str("-42").unwrap();
        assert!(matches!(prog.ops.as_ref(), [Opcode::PushInt(-42)]));
    }

    #[test]
    fn fusion_topn_opcode() {
        use crate::vm::{Compiler, Opcode};
        let prog = Compiler::compile_str("$.nums.sort()[0:3]").unwrap();
        let has = prog.ops.iter().any(|o| matches!(o, Opcode::TopN { n: 3, asc: true }));
        assert!(has, "sort + [0:n] should fuse to TopN");
    }

    #[test]
    fn fusion_topn_semantics() {
        let doc = json!({"nums": [5, 3, 1, 4, 2, 9, 7]});
        let r = query("$.nums.sort()[0:3]", &doc).unwrap();
        assert_eq!(r, json!([1, 2, 3]));
    }

    #[test]
    fn analysis_monotonicity_sort() {
        use crate::analysis::{infer_monotonicity, Monotonicity};
        use crate::vm::Compiler;
        let p = Compiler::compile_str("$.x.sort()").unwrap();
        assert_eq!(infer_monotonicity(&p), Monotonicity::Asc);
        let p = Compiler::compile_str("$.x.sort().reverse()").unwrap();
        assert_eq!(infer_monotonicity(&p), Monotonicity::Desc);
    }

    #[test]
    fn analysis_cost_nonzero() {
        use crate::analysis::program_cost;
        use crate::vm::Compiler;
        let cheap = Compiler::compile_str("42").unwrap();
        let expensive = Compiler::compile_str("$.books.filter(@.price > 10).map(@.title).sort()").unwrap();
        assert!(program_cost(&cheap) < program_cost(&expensive));
    }

    #[test]
    fn analysis_selectivity_score() {
        use crate::analysis::selectivity_score;
        use crate::parser::parse;
        let eq = parse("x == 1").unwrap();
        let lt = parse("x < 1").unwrap();
        let t  = parse("true").unwrap();
        assert!(selectivity_score(&eq) < selectivity_score(&lt));
        assert!(selectivity_score(&lt) < selectivity_score(&t));
    }

    #[test]
    fn analysis_escapes_doc() {
        use crate::analysis::escapes_doc;
        use crate::vm::Compiler;
        let p = Compiler::compile_str("42").unwrap();
        assert!(!escapes_doc(&p));
        let p = Compiler::compile_str("$.x").unwrap();
        assert!(escapes_doc(&p));
    }

    #[test]
    fn fusion_map_sum_opcode() {
        use crate::vm::{Compiler, Opcode};
        let prog = Compiler::compile_str("$.books.map(@.price).sum()").unwrap();
        let has = prog.ops.iter().any(|o| matches!(o, Opcode::MapSum(_)));
        assert!(has, "map+sum should fuse to MapSum");
    }

    #[test]
    fn fusion_map_avg_opcode() {
        use crate::vm::{Compiler, Opcode};
        let prog = Compiler::compile_str("$.books.map(@.price).avg()").unwrap();
        let has = prog.ops.iter().any(|o| matches!(o, Opcode::MapAvg(_)));
        assert!(has, "map+avg should fuse to MapAvg");
    }

    #[test]
    fn fusion_filter_first_opcode() {
        use crate::vm::{Compiler, Opcode};
        let prog = Compiler::compile_str("$.books.filter(@.price > 10).first()").unwrap();
        let has = prog.ops.iter().any(|o| matches!(o, Opcode::FindFirst(_)));
        assert!(has, "filter+first should fuse to FindFirst");
    }

    #[test]
    fn fusion_filter_first_semantics() {
        let doc = books();
        let fused = query("$.store.books.filter(@.price > 10).first()", &doc).unwrap();
        let plain = query("$.store.books.filter(@.price > 10) | first()", &doc).unwrap();
        assert_eq!(fused, plain);
    }

    #[test]
    fn fusion_filter_map_sum_opcode() {
        use crate::vm::{Compiler, Opcode};
        let prog = Compiler::compile_str(
            "$.books.filter(@.price > 10).map(@.price).sum()"
        ).unwrap();
        let has = prog.ops.iter().any(|o| matches!(o, Opcode::FilterMapSum { .. }));
        assert!(has, "filter+map+sum should fuse to FilterMapSum");
    }

    #[test]
    fn fusion_filter_map_avg_opcode() {
        use crate::vm::{Compiler, Opcode};
        let prog = Compiler::compile_str(
            "$.books.filter(@.price > 10).map(@.price).avg()"
        ).unwrap();
        let has = prog.ops.iter().any(|o| matches!(o, Opcode::FilterMapAvg { .. }));
        assert!(has, "filter+map+avg should fuse to FilterMapAvg");
    }

    #[test]
    fn fusion_filter_map_sum_semantics() {
        let doc = books();
        let fused = query(
            "$.store.books.filter(@.price > 10).map(@.price).sum()", &doc
        ).unwrap();
        let plain = query(
            "$.store.books.filter(@.price > 10).sum(price)", &doc
        ).unwrap();
        assert_eq!(fused, plain);
    }

    #[test]
    fn fusion_filter_map_avg_semantics() {
        let doc = books();
        let fused = query(
            "$.store.books.filter(@.price > 10).map(@.price).avg()", &doc
        ).unwrap();
        let plain = query(
            "$.store.books.filter(@.price > 10).avg(price)", &doc
        ).unwrap();
        assert_eq!(fused, plain);
    }

    #[test]
    fn fusion_map_first_opcode() {
        use crate::vm::{Compiler, Opcode};
        let prog = Compiler::compile_str("$.books.map(@.price).first()").unwrap();
        let has = prog.ops.iter().any(|o| matches!(o, Opcode::MapFirst(_)));
        assert!(has, "map+first should fuse to MapFirst");
    }

    #[test]
    fn fusion_map_last_opcode() {
        use crate::vm::{Compiler, Opcode};
        let prog = Compiler::compile_str("$.books.map(@.price).last()").unwrap();
        let has = prog.ops.iter().any(|o| matches!(o, Opcode::MapLast(_)));
        assert!(has, "map+last should fuse to MapLast");
    }

    #[test]
    fn fusion_map_first_last_semantics() {
        let doc = books();
        let f = query("$.store.books.map(@.price).first()", &doc).unwrap();
        let l = query("$.store.books.map(@.price).last()",  &doc).unwrap();
        let all = query("$.store.books.map(@.price)",       &doc).unwrap();
        let arr = all.as_array().unwrap();
        assert_eq!(f, arr[0]);
        assert_eq!(l, arr[arr.len() - 1]);

        let empty_doc: serde_json::Value = serde_json::from_str(r#"{"xs":[]}"#).unwrap();
        let f_empty = query("$.xs.map(@.price).first()", &empty_doc).unwrap();
        assert!(f_empty.is_null());
    }

    #[test]
    fn fusion_filter_map_first_opcode() {
        use crate::vm::{Compiler, Opcode};
        let prog = Compiler::compile_str(
            "$.books.filter(@.price > 10).map(@.title).first()"
        ).unwrap();
        let has = prog.ops.iter().any(|o| matches!(o, Opcode::FilterMapFirst { .. }));
        assert!(has, "filter+map+first should fuse to FilterMapFirst");
    }

    #[test]
    fn fusion_filter_map_first_semantics() {
        let doc = books();
        let fused = query(
            "$.store.books.filter(@.price > 10).map(@.title).first()",
            &doc,
        ).unwrap();
        let plain = query(
            "$.store.books.filter(@.price > 10).map(@.title)",
            &doc,
        ).unwrap();
        let arr = plain.as_array().unwrap();
        if arr.is_empty() {
            assert!(fused.is_null());
        } else {
            assert_eq!(fused, arr[0]);
        }

        let empty_doc: serde_json::Value = serde_json::from_str(r#"{"xs":[]}"#).unwrap();
        let e = query("$.xs.filter(@.price > 0).map(@.title).first()", &empty_doc).unwrap();
        assert!(e.is_null());
    }

    #[test]
    fn const_fold_string_concat_and_cmp() {
        use crate::vm::{Compiler, Opcode};
        let p1 = Compiler::compile_str(r#"$ | "a" + "bc""#).unwrap();
        assert!(p1.ops.iter().any(|o| matches!(o, Opcode::PushStr(s) if s.as_ref() == "abc")),
                "\"a\" + \"bc\" should fold to PushStr(\"abc\"): {:?}", p1.ops);

        let p2 = Compiler::compile_str(r#"$ | "a" < "b""#).unwrap();
        assert!(p2.ops.iter().any(|o| matches!(o, Opcode::PushBool(true))),
                "\"a\" < \"b\" should fold to PushBool(true): {:?}", p2.ops);

        let p3 = Compiler::compile_str(r#"$ | "zz" >= "aa""#).unwrap();
        assert!(p3.ops.iter().any(|o| matches!(o, Opcode::PushBool(true))),
                "\"zz\" >= \"aa\" should fold: {:?}", p3.ops);
    }

    #[test]
    fn fusion_filter_last_opcode() {
        use crate::vm::{Compiler, Opcode};
        let prog = Compiler::compile_str("$.books.filter(@.price > 10).last()").unwrap();
        let has = prog.ops.iter().any(|o| matches!(o, Opcode::FilterLast { .. }));
        assert!(has, "filter+last should fuse to FilterLast: {:?}", prog.ops);
    }

    #[test]
    fn fusion_filter_last_semantics() {
        let doc = books();
        let fused = query("$.store.books.filter(@.price > 10).last()", &doc).unwrap();
        let plain = query("$.store.books.filter(@.price > 10)",        &doc).unwrap();
        let arr = plain.as_array().unwrap();
        if arr.is_empty() {
            assert!(fused.is_null());
        } else {
            assert_eq!(fused, arr[arr.len() - 1]);
        }

        let empty_doc: serde_json::Value = serde_json::from_str(r#"{"xs":[]}"#).unwrap();
        let e = query("$.xs.filter(@.price > 0).last()", &empty_doc).unwrap();
        assert!(e.is_null());
    }

    #[test]
    fn fusion_sort_sort_idempotent_collapse() {
        use crate::vm::{Compiler, Opcode, BuiltinMethod};
        let prog = Compiler::compile_str("$.books.sort().sort().first()").unwrap();
        let sorts = prog.ops.iter().filter(|o| matches!(o,
            Opcode::CallMethod(c) if c.method == BuiltinMethod::Sort)).count();
        // Doubled sort collapses to one; sort()+first() then strength-reduces to min().
        assert_eq!(sorts, 0, "sort().sort() should collapse: {:?}", prog.ops);
    }

    #[test]
    fn fusion_unique_unique_collapse() {
        use crate::vm::{Compiler, Opcode, BuiltinMethod};
        let prog = Compiler::compile_str("$.items.unique().unique()").unwrap();
        let uniqs = prog.ops.iter().filter(|o| matches!(o,
            Opcode::CallMethod(c) if c.method == BuiltinMethod::Unique)).count();
        assert_eq!(uniqs, 1, "unique().unique() should collapse: {:?}", prog.ops);
    }

    #[test]
    fn fusion_reverse_reverse_dropped() {
        use crate::vm::{Compiler, Opcode, BuiltinMethod};
        let prog = Compiler::compile_str("$.items.reverse().reverse()").unwrap();
        let revs = prog.ops.iter().filter(|o| matches!(o,
            Opcode::CallMethod(c) if c.method == BuiltinMethod::Reverse)).count();
        assert_eq!(revs, 0, "reverse().reverse() should be dropped: {:?}", prog.ops);
    }

    #[test]
    fn fusion_idempotent_semantics() {
        let doc = json!({"xs": [3, 1, 2, 1, 3]});
        let a = query("$.xs.unique().unique()", &doc).unwrap();
        let b = query("$.xs.unique()", &doc).unwrap();
        assert_eq!(a, b);
        let c = query("$.xs.reverse().reverse()", &doc).unwrap();
        assert_eq!(c, json!([3, 1, 2, 1, 3]));
    }

    #[test]
    fn fusion_sort_sum_drops_sort() {
        use crate::vm::{Compiler, Opcode, BuiltinMethod};
        let prog = Compiler::compile_str("$.books.sort().sum()").unwrap();
        let has_sort = prog.ops.iter().any(|o| matches!(o,
            Opcode::CallMethod(c) if c.method == BuiltinMethod::Sort));
        assert!(!has_sort, "sort before sum should be strength-reduced away");
    }

    #[test]
    fn fusion_reverse_max_drops_reverse() {
        use crate::vm::{Compiler, Opcode, BuiltinMethod};
        let prog = Compiler::compile_str("$.books.reverse().max()").unwrap();
        let has_rev = prog.ops.iter().any(|o| matches!(o,
            Opcode::CallMethod(c) if c.method == BuiltinMethod::Reverse));
        assert!(!has_rev, "reverse before max should be strength-reduced away");
    }

    #[test]
    fn fusion_reorder_aggregate_semantics() {
        let doc = books();
        // `min` / `max` are exact — no FP summation order to worry about.
        let a = query("$.store.books.sort(price).min(price)", &doc).unwrap();
        let b = query("$.store.books.min(price)",             &doc).unwrap();
        assert_eq!(a, b);
        let c = query("$.store.books.reverse().max(price)",   &doc).unwrap();
        let d = query("$.store.books.max(price)",             &doc).unwrap();
        assert_eq!(c, d);
    }

    #[test]
    fn fusion_map_sum_semantics() {
        let doc = books();
        let r = query("$.store.books.map(@.price).sum()", &doc).unwrap();
        let plain = query("$.store.books.sum(price)", &doc).unwrap();
        assert_eq!(r, plain);
    }

    #[test]
    fn fusion_map_avg_semantics() {
        let doc = books();
        let r = query("$.store.books.map(@.price).avg()", &doc).unwrap();
        let plain = query("$.store.books.avg(price)", &doc).unwrap();
        assert_eq!(r, plain);
    }

    #[test]
    fn analysis_dedup_subprograms() {
        use crate::analysis::dedup_subprograms;
        use crate::vm::Compiler;
        // Two identical sub-exprs inside an array — after dedup they share Arc.
        let prog = Compiler::compile_str("[$.a.b + 1, $.a.b + 1]").unwrap();
        let deduped = dedup_subprograms(&prog);
        // Extract MakeArr sub-progs and confirm Arc identity.
        use crate::vm::Opcode;
        let arcs: Vec<_> = deduped.ops.iter().flat_map(|o| match o {
            Opcode::MakeArr(progs) => progs.iter().cloned().collect::<Vec<_>>(),
            _ => vec![],
        }).collect();
        assert_eq!(arcs.len(), 2);
        assert!(std::sync::Arc::ptr_eq(&arcs[0], &arcs[1]),
                "identical sub-progs should share Arc");
    }

    #[test]
    fn analysis_find_common_subexprs() {
        use crate::analysis::find_common_subexprs;
        use crate::vm::Compiler;
        let prog = Compiler::compile_str("[$.x.y, $.x.y, $.z]").unwrap();
        let cs = find_common_subexprs(&prog);
        assert!(!cs.is_empty(), "should find at least one repeated sub-program");
    }

    #[test]
    fn method_const_fold_str_len() {
        use crate::vm::{Compiler, Opcode};
        let prog = Compiler::compile_str("\"hello\".len()").unwrap();
        assert!(matches!(prog.ops.as_ref(), [Opcode::PushInt(5)]));
    }

    #[test]
    fn method_const_fold_str_upper() {
        use crate::vm::{Compiler, Opcode};
        let prog = Compiler::compile_str("\"hi\".upper()").unwrap();
        assert!(matches!(prog.ops.as_ref(), [Opcode::PushStr(s)] if s.as_ref() == "HI"));
    }

    #[test]
    fn method_const_fold_arr_len() {
        use crate::vm::{Compiler, Opcode};
        let prog = Compiler::compile_str("[1,2,3,4].len()").unwrap();
        assert!(matches!(prog.ops.as_ref(), [Opcode::PushInt(4)]));
    }

    #[test]
    fn analysis_expr_uses_ident() {
        use crate::analysis::expr_uses_ident;
        use crate::parser::parse;
        let e = parse("x + 1").unwrap();
        assert!(expr_uses_ident(&e, "x"));
        assert!(!expr_uses_ident(&e, "y"));
        let e = parse("let x = 10 in x + 1").unwrap();
        assert!(!expr_uses_ident(&e, "x"), "inner let shadows");
    }

    #[test]
    fn analysis_fold_kind_check_helper() {
        use crate::analysis::{fold_kind_check, VType};
        use crate::ast::KindType;
        assert_eq!(fold_kind_check(VType::Int, KindType::Number, false), Some(true));
        assert_eq!(fold_kind_check(VType::Str, KindType::Number, false), Some(false));
        assert_eq!(fold_kind_check(VType::Str, KindType::Str, true),    Some(false));
        assert_eq!(fold_kind_check(VType::Unknown, KindType::Str, false), None);
    }

    // ── Optimiser power tests: large queries exercising fusions ──────────────
    //
    // These tests drive the VM through deep pipelines that hit every
    // fusion pass (FilterMap, FilterFilter, MapMap, FilterCount,
    // FindFirst, FindOne, MapSum, MapAvg, TopN, MapFlatten,
    // FilterTakeWhile, FilterDropWhile, MapUnique) plus RootChain
    // fusion, constant folding, strength reduction, and CSE.  Each
    // asserts the optimized VM produces exactly the documented output.

    fn big_store() -> serde_json::Value {
        // 20-book store with heterogeneous data — prices, ratings,
        // multi-tag arrays, nested authors, nullable fields.
        json!({
            "store": {
                "books": [
                    {"id": 1,  "title": "Dune",              "price": 12.99, "rating": 4.8, "genre": "sci-fi",   "tags": ["sci-fi","classic"],    "author": {"name": "Frank Herbert",  "born": 1920}, "pages": 688},
                    {"id": 2,  "title": "Foundation",        "price":  9.99, "rating": 4.5, "genre": "sci-fi",   "tags": ["sci-fi","series"],     "author": {"name": "Isaac Asimov",   "born": 1920}, "pages": 255},
                    {"id": 3,  "title": "Neuromancer",       "price": 11.50, "rating": 4.2, "genre": "cyberpunk","tags": ["sci-fi","cyberpunk"],  "author": {"name": "William Gibson", "born": 1948}, "pages": 271},
                    {"id": 4,  "title": "1984",              "price":  7.99, "rating": 4.6, "genre": "dystopia", "tags": ["classic","dystopia"],  "author": {"name": "George Orwell",  "born": 1903}, "pages": 328},
                    {"id": 5,  "title": "Brave New World",   "price":  8.50, "rating": 4.3, "genre": "dystopia", "tags": ["classic","dystopia"],  "author": {"name": "Aldous Huxley",  "born": 1894}, "pages": 311},
                    {"id": 6,  "title": "Hyperion",          "price": 13.25, "rating": 4.7, "genre": "sci-fi",   "tags": ["sci-fi","epic"],       "author": {"name": "Dan Simmons",    "born": 1948}, "pages": 482},
                    {"id": 7,  "title": "Snow Crash",        "price": 10.50, "rating": 4.1, "genre": "cyberpunk","tags": ["sci-fi","cyberpunk"],  "author": {"name": "Neal Stephenson","born": 1959}, "pages": 470},
                    {"id": 8,  "title": "Fahrenheit 451",    "price":  6.99, "rating": 4.4, "genre": "dystopia", "tags": ["classic","dystopia"],  "author": {"name": "Ray Bradbury",   "born": 1920}, "pages": 249},
                    {"id": 9,  "title": "Ender's Game",      "price":  8.75, "rating": 4.6, "genre": "sci-fi",   "tags": ["sci-fi","military"],   "author": {"name": "Orson Scott Card","born": 1951},"pages": 324},
                    {"id": 10, "title": "The Left Hand",     "price":  9.25, "rating": 4.2, "genre": "sci-fi",   "tags": ["sci-fi","feminist"],   "author": {"name": "Ursula K. Le Guin","born":1929},"pages": 304},
                    {"id": 11, "title": "A Scanner Darkly",  "price":  8.00, "rating": 4.0, "genre": "sci-fi",   "tags": ["sci-fi","philosophy"], "author": {"name": "Philip K. Dick", "born": 1928}, "pages": 280},
                    {"id": 12, "title": "Gateway",           "price":  7.50, "rating": 4.1, "genre": "sci-fi",   "tags": ["sci-fi","classic"],    "author": {"name": "Frederik Pohl",  "born": 1919}, "pages": 313},
                    {"id": 13, "title": "Stranger",          "price":  9.00, "rating": 4.3, "genre": "sci-fi",   "tags": ["sci-fi","classic"],    "author": {"name": "Robert Heinlein","born": 1907}, "pages": 438},
                    {"id": 14, "title": "Rendezvous",        "price": 10.00, "rating": 4.5, "genre": "sci-fi",   "tags": ["sci-fi","classic"],    "author": {"name": "Arthur C. Clarke","born": 1917},"pages": 304},
                    {"id": 15, "title": "Solaris",           "price":  8.25, "rating": 4.2, "genre": "sci-fi",   "tags": ["sci-fi","philosophy"], "author": {"name": "Stanisław Lem",  "born": 1921}, "pages": 204},
                    {"id": 16, "title": "The Road",          "price":  9.75, "rating": 4.4, "genre": "dystopia", "tags": ["literary","dystopia"], "author": {"name": "Cormac McCarthy","born": 1933}, "pages": 287},
                    {"id": 17, "title": "Never Let Me Go",   "price":  8.50, "rating": 4.3, "genre": "dystopia", "tags": ["literary","dystopia"], "author": {"name": "Kazuo Ishiguro", "born": 1954}, "pages": 288},
                    {"id": 18, "title": "Station Eleven",    "price": 11.00, "rating": 4.5, "genre": "dystopia", "tags": ["literary","dystopia"], "author": {"name": "Emily St. John", "born": 1979}, "pages": 333},
                    {"id": 19, "title": "The Martian",       "price": 12.00, "rating": 4.7, "genre": "sci-fi",   "tags": ["sci-fi","survival"],   "author": {"name": "Andy Weir",      "born": 1972}, "pages": 369},
                    {"id": 20, "title": "Project Hail Mary", "price": 14.50, "rating": 4.9, "genre": "sci-fi",   "tags": ["sci-fi","survival"],   "author": {"name": "Andy Weir",      "born": 1972}, "pages": 496},
                ]
            }
        })
    }

    #[test]
    fn optimized_deep_filter_map_map_fusion() {
        // Pipeline: filter + map + map chains fuse pairwise into
        // FilterMap then MapMap, eliminating intermediate Vec<Val>
        // allocations.  Also exercises strength-reduction: sort() + [0:3]
        // folds to TopN.
        let doc = big_store();
        let q = "$.store.books \
                 .filter(price >= 8.0 and price <= 12.0 and rating >= 4.2) \
                 .map({title: title, cost: price, score: rating}) \
                 .map({label: title, gross: cost}) \
                 .sort(gross)[0:3]";
        let r = query(q, &doc).unwrap();
        let arr = r.as_array().unwrap();
        assert_eq!(arr.len(), 3);
        // Lowest three prices in the allowed band, sorted asc.
        let grosses: Vec<f64> = arr.iter().map(|v| v["gross"].as_f64().unwrap()).collect();
        let mut sorted = grosses.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert_eq!(grosses, sorted);
        // All within band.
        assert!(grosses.iter().all(|&g| g >= 8.0 && g <= 12.0));
    }

    #[test]
    fn optimized_filter_sum_fusion_with_kind_check() {
        // Kind-check const-fold + MapSum fusion.
        // price kind number is a guaranteed true for all rows, so the
        // kind-check predicate folds out, leaving filter(...) + sum(...).
        let doc = big_store();
        let q = "$.store.books \
                 .filter(price kind number and genre == \"sci-fi\") \
                 .sum(price)";
        let r = query(q, &doc).unwrap();
        let total = r.as_f64().unwrap();
        // Hand-tally of the 12 sci-fi prices (Neuromancer + Snow Crash
        // are cyberpunk, not sci-fi).
        let expected = 12.99 + 9.99 + 13.25 + 8.75 + 9.25 + 8.00
                     + 7.50 + 9.00 + 10.00 + 8.25 + 12.00 + 14.50;
        assert!((total - expected).abs() < 0.001, "got {} want {}", total, expected);
    }

    #[test]
    fn optimized_nested_let_with_cse_and_avg() {
        // let-binding + reuse triggers interprocedural type inference
        // and let-init is pure → one emit (dead-let elimination when
        // body wouldn't use it, which is not the case here).  Also
        // MapAvg fusion via filter(...).avg(...).
        let doc = big_store();
        let q = "let sci = $.store.books.filter(genre == \"sci-fi\") in \
                 {\
                    count: sci.len(), \
                    avg_price:  sci.avg(price), \
                    avg_rating: sci.avg(rating), \
                    top_rated:  sci.sort(rating).reverse()[0:3].map(title)\
                 }";
        let r = query(q, &doc).unwrap();
        assert_eq!(r["count"], json!(12));
        assert!(r["avg_price"].as_f64().unwrap() > 8.0);
        assert!(r["avg_rating"].as_f64().unwrap() > 4.0);
        let top = r["top_rated"].as_array().unwrap();
        assert_eq!(top.len(), 3);
        // Project Hail Mary (4.9) should be first.
        assert_eq!(top[0], json!("Project Hail Mary"));
    }

    #[test]
    fn optimized_find_quantifier_fusion_short_circuit() {
        // filter(...).first() → FindFirst; short-circuits at first match.
        // Combined with AndOp operand reordering (selectivity): the
        // cheaper predicate (id == 19) should be evaluated first even
        // though it appears second in source.
        let doc = big_store();
        let q = "$.store.books.filter(rating > 4.5 and id == 19).first()";
        let r = query(q, &doc).unwrap();
        assert_eq!(r["title"], json!("The Martian"));
        assert_eq!(r["id"], json!(19));
    }

    #[test]
    fn optimized_group_then_aggregate_complex_reshape() {
        // Large pipeline: filter → group_by → transform_values → sort
        // by computed key.  Demonstrates object-return from map over
        // grouped buckets plus RootChain fusion inside the group key.
        let doc = big_store();
        let q = "$.store.books \
                 .filter(rating >= 4.0) \
                 .group_by(genre) \
                 .entries() \
                 .map({genre: @[0], count: @[1].len(), avg_price: @[1].avg(price)}) \
                 .sort(avg_price) \
                 .reverse()";
        let r = query(q, &doc).unwrap();
        let arr = r.as_array().unwrap();
        assert!(arr.len() >= 3); // sci-fi, dystopia, cyberpunk
        let genres: Vec<&str> = arr.iter().map(|v| v["genre"].as_str().unwrap()).collect();
        // All distinct.
        let mut u = genres.clone(); u.sort(); u.dedup();
        assert_eq!(u.len(), genres.len());
    }

    #[test]
    fn optimized_map_flatten_fusion_with_unique() {
        // map(f).flatten() → MapFlatten (single-pass concat).
        // Then map(f).unique() → MapUnique (streaming dedup).
        let doc = big_store();
        let q = "$.store.books.map(tags).flatten().unique().sort()";
        let r = query(q, &doc).unwrap();
        let tags = r.as_array().unwrap();
        // All distinct and sorted.
        let strs: Vec<&str> = tags.iter().map(|v| v.as_str().unwrap()).collect();
        let mut s = strs.clone(); s.sort(); s.dedup();
        assert_eq!(s, strs);
        // Must include well-known tags.
        assert!(strs.contains(&"sci-fi"));
        assert!(strs.contains(&"dystopia"));
        assert!(strs.contains(&"cyberpunk"));
    }

    #[test]
    fn optimized_filter_take_while_fusion() {
        // filter(p).takewhile(q) → FilterTakeWhile — scans until q fails.
        // Proves early-out behaviour: later high-priced books must NOT
        // leak through even though they pass the filter.
        let doc = big_store();
        // Books in order, priced > 5; take while price < 12.
        let q = "$.store.books.filter(price > 5.0).takewhile(price < 12.0).map(title)";
        let r = query(q, &doc).unwrap();
        let titles = r.as_array().unwrap();
        // Dune (12.99) breaks the scan immediately — result should be empty.
        // Actually Dune is first and its price > 12, so scan stops at 0.
        assert_eq!(titles.len(), 0);
    }

    #[test]
    fn optimized_deep_chain_with_comprehension_and_fstring() {
        // Very large expression: comprehension over filtered projection
        // with nested filter + f-string.  Exercises RootChain fusion
        // at multiple sites, list comprehension lowering, and string
        // interpolation opcodes.
        let doc = big_store();
        let q = "[f\"{b.title} (${b.price})\" \
                 for b in $.store.books \
                 if b.rating >= 4.5 and b.genre == \"sci-fi\" \
                 and b.author.born >= 1940]";
        let r = query(q, &doc).unwrap();
        let items = r.as_array().unwrap();
        // Hyperion, Ender's Game, The Martian, Project Hail Mary.
        assert!(items.len() >= 3);
        for s in items {
            let t = s.as_str().unwrap();
            assert!(t.contains("$"));
        }
    }

    #[test]
    fn optimized_let_chained_pipelines_with_aggregation() {
        // Three-level let chain with reuse.  Tests liveness: `books` is
        // used throughout, `cheap` only in one branch.  Body returns a
        // large reshape of aggregates.
        let doc = big_store();
        let q = "let books = $.store.books in \
                 let cheap = books.filter(price < 10.0) in \
                 let expensive = books.filter(price >= 10.0) in \
                 {\
                    total: books.len(), \
                    cheap_count: cheap.len(), \
                    expensive_count: expensive.len(), \
                    cheap_avg: cheap.avg(price), \
                    expensive_avg: expensive.avg(price), \
                    delta: expensive.avg(price) - cheap.avg(price), \
                    price_range: books.max(price) - books.min(price), \
                    top_author: books.sort(rating).reverse()[0].author.name\
                 }";
        let r = query(q, &doc).unwrap();
        assert_eq!(r["total"], json!(20));
        assert!(r["cheap_count"].as_i64().unwrap() > 0);
        assert!(r["expensive_count"].as_i64().unwrap() > 0);
        assert!(r["delta"].as_f64().unwrap() > 0.0);
        assert!(r["price_range"].as_f64().unwrap() > 5.0);
        // The book with max rating (4.9) is Project Hail Mary by Andy Weir.
        assert_eq!(r["top_author"], json!("Andy Weir"));
    }

    #[test]
    fn optimized_const_fold_across_arithmetic_and_comparisons() {
        // Every predicate is trivially-constant-foldable but wrapped
        // around a real filter path.  Const-fold must eliminate the
        // dead branches so the filter reduces to the runtime predicate.
        let doc = big_store();
        let q = "$.store.books \
                 .filter((1 + 2) * 3 == 9 and not (5 < 3) and price > 11.0) \
                 .map(title) \
                 .sort()";
        let r = query(q, &doc).unwrap();
        let titles: Vec<String> = r.as_array().unwrap().iter()
            .map(|v| v.as_str().unwrap().to_string()).collect();
        assert!(titles.iter().any(|t| t == "Dune"));
        assert!(titles.iter().any(|t| t == "Project Hail Mary"));
    }

    // Fusion benchmark — run with:
    //   cargo test --release tests::tests::bench_fusion_vs_naive -- --ignored --nocapture
    // Measures wall time for a representative pipeline with every pass
    // enabled vs every pass disabled over a 2000-book document.  Use
    // --release for realistic numbers; debug builds dominate with
    // bounds-check / iteration overhead that dwarfs fusion wins.
    #[test]
    #[ignore]
    fn bench_fusion_vs_naive() {
        use crate::vm::{VM, PassConfig};
        use std::time::Instant;

        // Synthesize a large store (2000 books).
        let mut books = Vec::with_capacity(2000);
        let genres = ["sci-fi","dystopia","cyberpunk","classic"];
        for i in 0..2000 {
            let g = genres[i % 4];
            books.push(json!({
                "id": i,
                "title": format!("Book {}", i),
                "price": (i % 50) as f64 + 5.0,
                "rating": ((i * 7) % 50) as f64 / 10.0,
                "genre": g,
                "tags":  ["a","b","c","d"],
                "author": {"name": format!("Author {}", i % 100), "born": 1900 + (i % 120)},
            }));
        }
        let doc = json!({"store": {"books": books}});

        let pipelines = &[
            "$.store.books.filter(price > 20 and rating > 3.5).map({t: title, p: price}).map({label: t, cost: p}).sort(cost)[0:10]",
            "$.store.books.filter(genre == \"sci-fi\").sum(price)",
            "$.store.books.map(tags).flatten().unique().sort()",
            "$.store.books.filter(price > 10).first()",
            "$.store.books.group_by(genre).entries().map({g: @[0], n: @[1].len(), avg: @[1].avg(price)})",
        ];

        let iters = 50;
        for q in pipelines {
            // Fused
            let mut vm = VM::new();
            vm.set_pass_config(PassConfig::default());
            let start = Instant::now();
            for _ in 0..iters { let _ = vm.run_str(q, &doc).unwrap(); }
            let fused = start.elapsed();

            // Naive
            let mut vm = VM::new();
            vm.set_pass_config(PassConfig::none());
            let start = Instant::now();
            for _ in 0..iters { let _ = vm.run_str(q, &doc).unwrap(); }
            let naive = start.elapsed();

            let ratio = naive.as_secs_f64() / fused.as_secs_f64();
            println!("[bench] iters={:<3} fused={:>8.2?} naive={:>8.2?}  speedup={:.2}x  q={}",
                     iters, fused, naive, ratio, q);
        }
    }

    #[test]
    fn pass_config_cache_isolation() {
        use crate::vm::{VM, PassConfig};
        let mut vm = VM::new();
        let doc = big_store();
        let q = "$.store.books.filter(price > 10).map(title).sort()";

        // Default config — all passes on.
        let r1 = vm.run_str(q, &doc).unwrap();
        let (n1, _) = vm.cache_stats();
        assert!(n1 >= 1);

        // Disable every pass.  Different config-hash → separate cache slot.
        vm.set_pass_config(PassConfig::none());
        let r2 = vm.run_str(q, &doc).unwrap();
        let (n2, _) = vm.cache_stats();
        assert_eq!(n2, n1 + 1, "separate cache entry per config");

        // Semantics preserved regardless of passes.
        assert_eq!(r1, r2);
    }

    #[test]
    fn lru_compile_cache_evicts_oldest() {
        use crate::vm::VM;
        let mut vm = VM::with_capacity(2, 16);
        let doc = json!({"a": 1, "b": 2, "c": 3});
        let _ = vm.run_str("$.a", &doc).unwrap();
        let _ = vm.run_str("$.b", &doc).unwrap();
        assert_eq!(vm.cache_stats().0, 2);
        // Touch "$.a" — "$.b" becomes LRU.
        let _ = vm.run_str("$.a", &doc).unwrap();
        let _ = vm.run_str("$.c", &doc).unwrap();
        assert_eq!(vm.cache_stats().0, 2, "cap enforced");
        // "$.a" still cached.
        let _ = vm.run_str("$.a", &doc).unwrap();
        assert_eq!(vm.cache_stats().0, 2);
        // "$.b" was evicted; re-running reinserts but stays capped.
        let _ = vm.run_str("$.b", &doc).unwrap();
        assert_eq!(vm.cache_stats().0, 2);
    }

    #[test]
    fn optimized_equi_join_hash_probe() {
        // Fused EquiJoin: books joined with author_detail by author_name.
        let doc = json!({
            "books": [
                {"title": "Dune",       "author_name": "Frank Herbert"},
                {"title": "Foundation", "author_name": "Isaac Asimov"},
                {"title": "1984",       "author_name": "George Orwell"},
                {"title": "Unknown",    "author_name": "No Match"},
            ],
            "authors": [
                {"name": "Frank Herbert", "nationality": "US"},
                {"name": "Isaac Asimov",  "nationality": "US"},
                {"name": "George Orwell", "nationality": "UK"},
            ]
        });
        let q = "$.books.equi_join($.authors, \"author_name\", \"name\")";
        let r = query(q, &doc).unwrap();
        let rows = r.as_array().unwrap();
        assert_eq!(rows.len(), 3); // "Unknown" has no match
        for row in rows {
            assert!(row.get("title").is_some());
            assert!(row.get("nationality").is_some());
            assert!(row.get("name").is_some());
        }
    }

    #[test]
    fn optimized_pipeline_stress_many_stages() {
        // Fifteen-stage pipeline, many of which participate in pairwise
        // fusion.  Result asserts concrete shape to prove every stage
        // ran in the right order after optimisation.
        let doc = big_store();
        let q = "$.store.books \
                 .filter(rating >= 4.0) \
                 .filter(price < 15.0) \
                 .map({t: title, p: price, r: rating, g: genre, b: author.born}) \
                 .filter(b >= 1900) \
                 .map({title: t, price: p, score: r * 20, genre: g, era: b}) \
                 .filter(score >= 85) \
                 .sort(price) \
                 .reverse() \
                 .map({title, price}) \
                 [0:5]";
        let r = query(q, &doc).unwrap();
        let arr = r.as_array().unwrap();
        assert_eq!(arr.len(), 5);
        // Must all have title/price keys, price descending.
        let mut last = f64::MAX;
        for it in arr {
            assert!(it.get("title").is_some());
            let p = it["price"].as_f64().unwrap();
            assert!(p <= last, "not sorted descending: {} > {}", p, last);
            last = p;
        }
    }

    // ── Tier-1A/1B syntax refinements ─────────────────────────────────────────

    #[test]
    fn pipe_alias_long() {
        let doc = books();
        let a = query("$.store.books | count()", &doc).unwrap();
        let b = query("$.store.books |> count()", &doc).unwrap();
        assert_eq!(a, b);
        assert_eq!(a, json!(4));
    }

    #[test]
    fn coalesce_double_q() {
        let doc = json!({"a": null, "b": 5});
        let a = query("$.a ?| $.b", &doc).unwrap();
        let b = query("$.a ?? $.b", &doc).unwrap();
        assert_eq!(a, b);
        assert_eq!(a, json!(5));
    }

    #[test]
    fn coalesce_double_q_chain() {
        let doc = json!({"a": null, "b": null, "c": 7});
        let r = query("$.a ?? $.b ?? $.c", &doc).unwrap();
        assert_eq!(r, json!(7));
    }

    #[test]
    fn is_kind_alias() {
        let doc = json!({"x": 42});
        let a = query("$.x kind number", &doc).unwrap();
        let b = query("$.x is number", &doc).unwrap();
        assert_eq!(a, b);
        assert_eq!(a, json!(true));
    }

    #[test]
    fn is_not_kind_alias() {
        let doc = json!({"x": "hello"});
        let a = query("$.x kind not number", &doc).unwrap();
        let b = query("$.x is not number", &doc).unwrap();
        assert_eq!(a, b);
        assert_eq!(a, json!(true));
    }

    #[test]
    fn multi_binding_let() {
        let doc = books();
        let r = query("let a = 2, b = 3 in a + b", &doc).unwrap();
        assert_eq!(r, json!(5));
    }

    #[test]
    fn multi_binding_let_nested_ref() {
        let doc = books();
        // Second binding may refer to first (nested desugar).
        let r = query("let a = 10, b = a * 2 in b", &doc).unwrap();
        assert_eq!(r, json!(20));
    }

    #[test]
    fn arrow_lambda_single_param() {
        let doc = books();
        let r = query("$.store.books.map(b => b.price)", &doc).unwrap();
        let arr = r.as_array().unwrap();
        assert_eq!(arr.len(), 4);
        assert_eq!(arr[0], json!(12.99));
    }

    #[test]
    fn arrow_lambda_paren_params() {
        let doc = json!({"nums": [1,2,3,4]});
        let r = query("$.nums.map((x) => x * 2)", &doc).unwrap();
        assert_eq!(r, json!([2,4,6,8]));
    }

    #[test]
    fn arrow_lambda_multi_param() {
        // Sort supports a 2-param lambda comparator `lambda a,b: a<b`.
        // Test arrow-form `(a,b) => a < b` desugars identically.
        let doc = json!({"nums": [3, 1, 4, 1, 5, 9, 2, 6]});
        let r = query("$.nums.sort((a, b) => a < b)", &doc).unwrap();
        assert_eq!(r, json!([1, 1, 2, 3, 4, 5, 6, 9]));
    }

    #[test]
    fn cast_int_from_str() {
        let doc = json!({"s": "42"});
        let r = query("$.s as int", &doc).unwrap();
        assert_eq!(r, json!(42));
    }

    #[test]
    fn cast_float_from_int() {
        let doc = json!({"n": 3});
        let r = query("$.n as float", &doc).unwrap();
        assert_eq!(r, json!(3.0));
    }

    #[test]
    fn cast_str_from_number() {
        let doc = json!({"n": 42});
        let r = query("$.n as string", &doc).unwrap();
        assert_eq!(r, json!("42"));
    }

    #[test]
    fn cast_bool_from_int() {
        let doc = json!({"n": 1});
        let r = query("$.n as bool", &doc).unwrap();
        assert_eq!(r, json!(true));
    }

    #[test]
    fn cast_chain_with_arithmetic() {
        let doc = json!({"s": "10"});
        // `as` tighter than *, so `$.s as int * 2` == `($.s as int) * 2` == 20.
        let r = query("$.s as int * 2", &doc).unwrap();
        assert_eq!(r, json!(20));
    }

    #[test]
    fn dyn_field_string_key() {
        let doc = json!({"user": {"name": "Alice", "age": 30}});
        // `.{expr}` picks a key dynamically — same semantics as [expr].
        let r = query("let k = \"name\" in $.user.{k}", &doc).unwrap();
        assert_eq!(r, json!("Alice"));
    }

    #[test]
    fn dyn_field_int_index() {
        let doc = json!({"items": [10, 20, 30]});
        let r = query("let i = 1 in $.items.{i}", &doc).unwrap();
        assert_eq!(r, json!(20));
    }

    #[test]
    fn dyn_field_computed_key() {
        let doc = json!({"prefix_name": "hello", "key": "name"});
        let r = query("$.{\"prefix_\" + $.key}", &doc).unwrap();
        assert_eq!(r, json!("hello"));
    }

    #[test]
    fn dyn_field_equivalent_to_bracket() {
        let doc = json!({"user": {"name": "Bob"}});
        let a = query("$.user[\"name\"]", &doc).unwrap();
        let b = query("$.user.{\"name\"}", &doc).unwrap();
        assert_eq!(a, b);
    }

    // ── Phase 2: [*] => map-into-shape template ───────────────────────────────

    #[test]
    fn map_shape_basic() {
        let doc = books();
        let a = query("$.store.books.map({title})", &doc).unwrap();
        let b = query("$.store.books[*] => {title}", &doc).unwrap();
        assert_eq!(a, b);
        assert_eq!(a.as_array().unwrap().len(), 4);
    }

    #[test]
    fn map_shape_with_guard() {
        let doc = books();
        let a = query("$.store.books.filter(price > 10).map({title, price})", &doc).unwrap();
        let b = query("$.store.books[* if price > 10] => {title, price}", &doc).unwrap();
        assert_eq!(a, b);
        let arr = b.as_array().unwrap();
        assert!(arr.len() >= 2);
        for it in arr {
            assert!(it.get("title").is_some());
            assert!(it.get("price").unwrap().as_f64().unwrap() > 10.0);
        }
    }

    #[test]
    fn map_shape_nested() {
        let doc = json!({
            "groups": [
                {"name": "A", "items": [{"n": 1}, {"n": 2}]},
                {"name": "B", "items": [{"n": 3}]},
            ]
        });
        let r = query("$.groups[*] => {name, ns: items[*] => n}", &doc).unwrap();
        assert_eq!(r, json!([
            {"name": "A", "ns": [1, 2]},
            {"name": "B", "ns": [3]},
        ]));
    }

    #[test]
    fn map_shape_projection_subset() {
        let doc = books();
        let r = query("$.store.books[*] => {title, price}", &doc).unwrap();
        let arr = r.as_array().unwrap();
        assert_eq!(arr.len(), 4);
        for it in arr {
            let obj = it.as_object().unwrap();
            assert_eq!(obj.len(), 2);
            assert!(obj.contains_key("title"));
            assert!(obj.contains_key("price"));
        }
    }

    // ── Phase 2: `when` conditional field ─────────────────────────────────────

    #[test]
    fn when_field_included_true() {
        let doc = json!({"name": "Alice", "email": "a@x.com", "verified": true});
        let r = query("{name, email: $.email when $.verified}", &doc).unwrap();
        assert_eq!(r, json!({"name": "Alice", "email": "a@x.com"}));
    }

    #[test]
    fn when_field_excluded_false() {
        let doc = json!({"name": "Alice", "email": "a@x.com", "verified": false});
        let r = query("{name, email: $.email when $.verified}", &doc).unwrap();
        assert_eq!(r, json!({"name": "Alice"}));
    }

    #[test]
    fn when_field_excluded_null_guard() {
        let doc = json!({"name": "Bob"});
        let r = query("{name, email: \"default\" when $.verified}", &doc).unwrap();
        assert_eq!(r, json!({"name": "Bob"}));
    }

    #[test]
    fn when_field_in_nested_shape() {
        let doc = json!({
            "users": [
                {"name": "A", "active": true,  "role": "admin"},
                {"name": "B", "active": false, "role": "user"},
            ]
        });
        // Inside map, `@` is current item — `role` (bare ident) reads from current.
        let r = query("$.users[*] => {name, role: role when active}", &doc).unwrap();
        assert_eq!(r, json!([
            {"name": "A", "role": "admin"},
            {"name": "B"},
        ]));
    }

    #[test]
    fn when_field_guard_uses_other_fields() {
        let doc = json!({"score": 85, "threshold": 70});
        let r = query("{grade: \"pass\" when score > threshold}", &doc).unwrap();
        assert_eq!(r, json!({"grade": "pass"}));
    }

    // ── Phase 2: `...**` deep-merge spread ────────────────────────────────────

    #[test]
    fn spread_deep_merges_nested_objects() {
        let doc = json!({
            "base": {"x": {"p": 1, "q": 2}, "y": 10},
            "over": {"x": {"q": 99, "r": 3}, "z": 20},
        });
        // Shallow spread would overwrite `x` entirely; deep-spread merges nested.
        let r = query("{...**$.base, ...**$.over}", &doc).unwrap();
        assert_eq!(r, json!({
            "x": {"p": 1, "q": 99, "r": 3},
            "y": 10,
            "z": 20,
        }));
    }

    #[test]
    fn spread_deep_vs_shallow() {
        let doc = json!({
            "base": {"x": {"p": 1}},
            "over": {"x": {"q": 2}},
        });
        let shallow = query("{...$.base, ...$.over}", &doc).unwrap();
        let deep    = query("{...**$.base, ...**$.over}", &doc).unwrap();
        // Shallow: x is just {q:2} (replaced). Deep: x is {p:1, q:2} (merged).
        assert_eq!(shallow, json!({"x": {"q": 2}}));
        assert_eq!(deep,    json!({"x": {"p": 1, "q": 2}}));
    }

    #[test]
    fn spread_deep_concatenates_arrays() {
        let doc = json!({
            "a": {"tags": ["one", "two"]},
            "b": {"tags": ["three"]},
        });
        let r = query("{...**$.a, ...**$.b}", &doc).unwrap();
        assert_eq!(r, json!({"tags": ["one", "two", "three"]}));
    }

    #[test]
    fn spread_deep_scalar_rhs_wins() {
        let doc = json!({
            "a": {"name": "Alice", "info": {"nested": 1}},
            "b": {"name": "Bob"},
        });
        let r = query("{...**$.a, ...**$.b}", &doc).unwrap();
        assert_eq!(r, json!({"name": "Bob", "info": {"nested": 1}}));
    }

    // ── Patch block ───────────────────────────────────────────────────────────

    #[test]
    fn patch_simple_field_replace() {
        let doc = json!({"name": "Alice", "age": 30});
        let r = query(r#"patch $ { name: "Bob" }"#, &doc).unwrap();
        assert_eq!(r, json!({"name": "Bob", "age": 30}));
    }

    #[test]
    fn patch_nested_field_replace() {
        let doc = json!({"user": {"name": "Alice", "age": 30}});
        let r = query(r#"patch $ { user.name: "Bob" }"#, &doc).unwrap();
        assert_eq!(r, json!({"user": {"name": "Bob", "age": 30}}));
    }

    #[test]
    fn patch_delete_field() {
        let doc = json!({"name": "Alice", "tmp": "remove-me", "age": 30});
        let r = query(r#"patch $ { tmp: DELETE }"#, &doc).unwrap();
        assert_eq!(r, json!({"name": "Alice", "age": 30}));
    }

    #[test]
    fn patch_add_new_field() {
        let doc = json!({"name": "Alice"});
        let r = query(r#"patch $ { age: 42 }"#, &doc).unwrap();
        assert_eq!(r, json!({"name": "Alice", "age": 42}));
    }

    #[test]
    fn patch_wildcard_array() {
        let doc = json!({"users": [
            {"name": "Alice", "seen": false},
            {"name": "Bob",   "seen": false},
        ]});
        let r = query(r#"patch $ { users[*].seen: true }"#, &doc).unwrap();
        assert_eq!(r, json!({"users": [
            {"name": "Alice", "seen": true},
            {"name": "Bob",   "seen": true},
        ]}));
    }

    #[test]
    fn patch_wildcard_filter() {
        let doc = json!({"users": [
            {"name": "Alice", "active": true,  "role": "user"},
            {"name": "Bob",   "active": false, "role": "user"},
            {"name": "Cara",  "active": true,  "role": "user"},
        ]});
        let r = query(r#"patch $ { users[* if active].role: "admin" }"#, &doc).unwrap();
        assert_eq!(r, json!({"users": [
            {"name": "Alice", "active": true,  "role": "admin"},
            {"name": "Bob",   "active": false, "role": "user"},
            {"name": "Cara",  "active": true,  "role": "admin"},
        ]}));
    }

    #[test]
    fn patch_uses_current_value() {
        let doc = json!({"users": [
            {"name": "Alice", "email": "ALICE@X"},
            {"name": "Bob",   "email": "BOB@X"},
        ]});
        let r = query(r#"patch $ { users[*].email: @.lower() }"#, &doc).unwrap();
        assert_eq!(r, json!({"users": [
            {"name": "Alice", "email": "alice@x"},
            {"name": "Bob",   "email": "bob@x"},
        ]}));
    }

    #[test]
    fn patch_conditional_when_truthy() {
        let doc = json!({"count": 5, "enabled": true});
        let r = query(r#"patch $ { count: @ + 1 when $.enabled }"#, &doc).unwrap();
        assert_eq!(r, json!({"count": 6, "enabled": true}));
    }

    #[test]
    fn patch_conditional_when_falsy_skips() {
        let doc = json!({"count": 5, "enabled": false});
        let r = query(r#"patch $ { count: @ + 1 when $.enabled }"#, &doc).unwrap();
        assert_eq!(r, json!({"count": 5, "enabled": false}));
    }

    #[test]
    fn patch_multiple_ops_in_order() {
        let doc = json!({"a": 1, "b": 2, "c": 3});
        let r = query(r#"patch $ { a: 10, b: DELETE, c: 30 }"#, &doc).unwrap();
        assert_eq!(r, json!({"a": 10, "c": 30}));
    }

    #[test]
    fn patch_index_access() {
        let doc = json!({"items": [10, 20, 30]});
        let r = query(r#"patch $ { items[1]: 99 }"#, &doc).unwrap();
        assert_eq!(r, json!({"items": [10, 99, 30]}));
    }

    #[test]
    fn patch_delete_from_wildcard() {
        let doc = json!({"users": [
            {"name": "Alice", "active": true},
            {"name": "Bob",   "active": false},
            {"name": "Cara",  "active": true},
        ]});
        let r = query(r#"patch $ { users[* if not active]: DELETE }"#, &doc).unwrap();
        assert_eq!(r, json!({"users": [
            {"name": "Alice", "active": true},
            {"name": "Cara",  "active": true},
        ]}));
    }

    #[test]
    fn patch_composes_pipe() {
        let doc = json!({"name": "Alice", "age": 30});
        let r = query(r#"patch $ { name: "Bob" } | @.name"#, &doc).unwrap();
        assert_eq!(r, json!("Bob"));
    }

    #[test]
    fn patch_composes_method_chain() {
        let doc = json!({"name": "Alice", "age": 30});
        let r = query(r#"patch $ { name: "Bob" }.keys()"#, &doc).unwrap();
        let mut keys = r.as_array().unwrap().clone();
        keys.sort_by(|a, b| a.as_str().unwrap().cmp(b.as_str().unwrap()));
        assert_eq!(keys, vec![json!("age"), json!("name")]);
    }

    #[test]
    fn patch_composes_nested_in_object() {
        let doc = json!({"name": "Alice"});
        let r = query(r#"{result: patch $ { name: "Bob" }}"#, &doc).unwrap();
        assert_eq!(r, json!({"result": {"name": "Bob"}}));
    }

    #[test]
    fn patch_composes_let_binding() {
        let doc = json!({"name": "Alice", "age": 30});
        let r = query(r#"let x = patch $ { name: "Bob" } in x.name"#, &doc).unwrap();
        assert_eq!(r, json!("Bob"));
    }

    #[test]
    fn patch_composes_nested_patch() {
        let doc = json!({"name": "Alice", "age": 30});
        let r = query(r#"patch (patch $ { name: "Bob" }) { age: 99 }"#, &doc).unwrap();
        assert_eq!(r, json!({"name": "Bob", "age": 99}));
    }

    #[test]
    fn patch_composes_inside_map() {
        let doc = json!({"users": [{"n": 1}, {"n": 2}, {"n": 3}]});
        let r = query(r#"$.users.map(patch @ { n: @ * 10 })"#, &doc).unwrap();
        assert_eq!(r, json!([{"n": 10}, {"n": 20}, {"n": 30}]));
    }

    #[test]
    fn patch_delete_mark_outside_patch_errors() {
        let doc = json!({});
        let r = query(r#"DELETE"#, &doc);
        assert!(r.is_err());
    }

    // ── Tier 1: aliases + unique_by + collect + deep_* ────────────────────────

    fn saas() -> serde_json::Value {
        json!({
            "org": "acme",
            "teams": [
                {
                    "name": "platform",
                    "members": [
                        {"email": "a@acme.io", "role": "lead"},
                        {"email": "b@acme.io", "role": "eng"}
                    ],
                    "projects": [
                        {"id": 1, "name": "api",     "tasks": [{"id": "t1", "status": "open"}, {"id": "t2", "status": "done"}]},
                        {"id": 2, "name": "runtime", "tasks": [{"id": "t3", "status": "open"}]}
                    ]
                },
                {
                    "name": "growth",
                    "members": [
                        {"email": "c@acme.io", "role": "lead"},
                        {"email": "a@acme.io", "role": "eng"}
                    ],
                    "projects": [
                        {"id": 3, "name": "funnel", "tasks": []}
                    ]
                }
            ],
            "billing": {
                "invoices": [
                    {"email": "acme-finance@acme.io", "total": 100}
                ]
            }
        })
    }

    #[test]
    fn tier1_find_alias() {
        let doc = books();
        let r = query(r#"$.store.books.find(price > 10).map(title)"#, &doc).unwrap();
        assert_eq!(r, json!(["Dune", "Neuromancer"]));
    }

    #[test]
    fn tier1_find_all_alias() {
        let doc = books();
        let r = query(r#"$.store.books.find_all(rating > 4.5).map(title)"#, &doc).unwrap();
        let titles = r.as_array().unwrap();
        assert!(titles.contains(&json!("Dune")));
        assert!(titles.contains(&json!("1984")));
    }

    #[test]
    fn tier1_unique_by_lambda() {
        let doc = saas();
        let r = query(
            r#"$.teams.flat_map(lambda t: t.members).unique_by(lambda m: m.email).map(email)"#,
            &doc,
        ).unwrap();
        let emails = r.as_array().unwrap();
        assert_eq!(emails.len(), 3);
        assert!(emails.contains(&json!("a@acme.io")));
        assert!(emails.contains(&json!("b@acme.io")));
        assert!(emails.contains(&json!("c@acme.io")));
    }

    #[test]
    fn tier1_collect_scalar() {
        let doc = json!({"x": 42});
        let r = query(r#"$.x.collect()"#, &doc).unwrap();
        assert_eq!(r, json!([42]));
    }

    #[test]
    fn tier1_collect_array_identity() {
        let doc = json!({"xs": [1,2,3]});
        let r = query(r#"$.xs.collect()"#, &doc).unwrap();
        assert_eq!(r, json!([1, 2, 3]));
    }

    #[test]
    fn tier1_collect_null_to_empty() {
        let doc = json!({"x": null});
        let r = query(r#"$.x.collect()"#, &doc).unwrap();
        assert_eq!(r, json!([]));
    }

    #[test]
    fn tier1_deep_shape_email_keys() {
        let doc = saas();
        // any object carrying `email` anywhere — members + invoices
        let r = query(r#"$.deep_shape({email})"#, &doc).unwrap();
        let arr = r.as_array().unwrap();
        // 4 members across teams + 1 invoice = 5
        assert_eq!(arr.len(), 5);
    }

    #[test]
    fn tier1_deep_like_role_lead() {
        let doc = saas();
        let r = query(r#"$.deep_like({role: "lead"})"#, &doc).unwrap();
        let arr = r.as_array().unwrap();
        assert_eq!(arr.len(), 2);
        for item in arr {
            assert_eq!(item.get("role").unwrap(), &json!("lead"));
        }
    }

    #[test]
    fn tier1_pick_ident_keys() {
        let doc = json!({"user": {"name": "Alice", "age": 30, "score": 85}});
        let r = query(r#"$.user.pick(name, age)"#, &doc).unwrap();
        assert_eq!(r, json!({"name": "Alice", "age": 30}));
    }

    #[test]
    fn tier1_pick_alias() {
        let doc = json!({"user": {"name": "Alice", "age": 30}});
        let r = query(r#"$.user.pick(name, years: age)"#, &doc).unwrap();
        assert_eq!(r, json!({"name": "Alice", "years": 30}));
    }

    #[test]
    fn tier1_pick_over_array() {
        let doc = saas();
        let r = query(r#"$.teams[0].members.pick(email)"#, &doc).unwrap();
        assert_eq!(r, json!([{"email": "a@acme.io"}, {"email": "b@acme.io"}]));
    }

    #[test]
    fn tier1_pick_alias_over_array() {
        let doc = saas();
        let r = query(r#"$.teams[0].members.pick(addr: email, role)"#, &doc).unwrap();
        assert_eq!(r, json!([
            {"addr": "a@acme.io", "role": "lead"},
            {"addr": "b@acme.io", "role": "eng"}
        ]));
    }

    #[test]
    fn tier1_deep_find_status_open() {
        let doc = saas();
        let r = query(r#"$.deep_find(@ kind object and status == "open")"#, &doc).unwrap();
        let arr = r.as_array().unwrap();
        assert_eq!(arr.len(), 2);
    }

    #[test]
    fn tier1_dotdot_find_sugar() {
        let doc = saas();
        let r = query(r#"$..find(@ kind object and status == "open")"#, &doc).unwrap();
        let arr = r.as_array().unwrap();
        assert_eq!(arr.len(), 2);
    }

    #[test]
    fn tier1_dotdot_shape_sugar() {
        let doc = saas();
        let r = query(r#"$..shape({email})"#, &doc).unwrap();
        let arr = r.as_array().unwrap();
        assert_eq!(arr.len(), 5);
    }

    #[test]
    fn tier1_dotdot_like_sugar() {
        let doc = saas();
        let r = query(r#"$..like({role: "lead"})"#, &doc).unwrap();
        let arr = r.as_array().unwrap();
        assert_eq!(arr.len(), 2);
    }

    // ── Tier 1: chain-style terminal writes ──────────────────────────────────

    #[test]
    fn tier1_chain_set_field() {
        let doc = json!({"user": {"name": "Alice", "age": 30}});
        let r = query(r#"$.user.name.set("Bob")"#, &doc).unwrap();
        assert_eq!(r, json!({"user": {"name": "Bob", "age": 30}}));
    }

    #[test]
    fn tier1_chain_set_deep() {
        let doc = saas();
        let r = query(r#"$.teams[0].projects[0].name.set("API")"#, &doc).unwrap();
        assert_eq!(r.pointer("/teams/0/projects/0/name").unwrap(), &json!("API"));
        // untouched siblings still present
        assert_eq!(r.pointer("/teams/0/projects/1/name").unwrap(), &json!("runtime"));
        assert_eq!(r.pointer("/org").unwrap(), &json!("acme"));
    }

    #[test]
    fn tier1_chain_modify_using_current() {
        let doc = json!({"counts": {"n": 5}});
        let r = query(r#"$.counts.n.modify(@ * 2)"#, &doc).unwrap();
        assert_eq!(r, json!({"counts": {"n": 10}}));
    }

    #[test]
    fn tier1_chain_delete_field() {
        let doc = json!({"user": {"name": "Alice", "age": 30}});
        let r = query(r#"$.user.age.delete()"#, &doc).unwrap();
        assert_eq!(r, json!({"user": {"name": "Alice"}}));
    }

    #[test]
    fn tier1_chain_unset_key() {
        let doc = json!({"user": {"name": "Alice", "age": 30}});
        let r = query(r#"$.user.unset("age")"#, &doc).unwrap();
        assert_eq!(r, json!({"user": {"name": "Alice"}}));
    }

    #[test]
    fn tier1_chain_set_subtree() {
        let doc = json!({"a": {"b": {"c": 1}}});
        let r = query(r#"$.a.b.set({x: 42})"#, &doc).unwrap();
        assert_eq!(r, json!({"a": {"b": {"x": 42}}}));
    }

    #[test]
    fn tier1_chain_descendant_set() {
        let doc = saas();
        // every `status` anywhere under the doc flips to "closed"
        let r = query(r#"$..status.set("closed")"#, &doc).unwrap();
        let statuses: Vec<&serde_json::Value> = r.pointer("/teams/0/projects/0/tasks").unwrap()
            .as_array().unwrap().iter()
            .map(|t| t.get("status").unwrap())
            .collect();
        assert!(statuses.iter().all(|s| *s == &json!("closed")));
    }

    #[test]
    fn tier1_chain_descendant_delete() {
        let doc = json!({"a": {"id": 1, "b": {"id": 2, "c": {"id": 3}}}});
        let r = query(r#"$..id.delete()"#, &doc).unwrap();
        assert_eq!(r, json!({"a": {"b": {"c": {}}}}));
    }

    #[test]
    fn tier1_chain_dyn_index() {
        let doc = json!({"xs": [10, 20, 30, 40], "i": 2});
        let r = query(r#"$.xs[$.i].set(99)"#, &doc).unwrap();
        assert_eq!(r.pointer("/xs").unwrap(), &json!([10, 20, 99, 40]));
    }

    #[test]
    fn tier1_chain_merge() {
        let doc = json!({"config": {"host": "a", "port": 80}});
        let r = query(r#"$.config.merge({port: 443, tls: true})"#, &doc).unwrap();
        assert_eq!(r, json!({"config": {"host": "a", "port": 443, "tls": true}}));
    }

    #[test]
    fn tier1_chain_deep_merge() {
        let doc = json!({"a": {"b": {"x": 1}}});
        let r = query(r#"$.a.deep_merge({b: {y: 2}})"#, &doc).unwrap();
        assert_eq!(r, json!({"a": {"b": {"x": 1, "y": 2}}}));
    }

    #[test]
    fn tier1_chain_modify_lambda() {
        let doc = json!({"counts": {"n": 5}});
        let r = query(r#"$.counts.n.modify(lambda x: x * 3)"#, &doc).unwrap();
        assert_eq!(r, json!({"counts": {"n": 15}}));
    }

    #[test]
    fn tier1_non_root_set_is_method_call() {
        // `.set` without `$` prefix is the old builtin: returns arg, ignoring recv
        let doc = json!({"x": 1});
        let r = query(r#"$.x | set(99)"#, &doc).unwrap();
        assert_eq!(r, json!(99));
    }

    #[test]
    fn tier1_descendant_still_works() {
        // `..field` (no parens) should still be a descendant lookup, not a deep method
        let doc = books();
        let r = query("$..title", &doc).unwrap();
        let titles = r.as_array().unwrap();
        assert!(titles.contains(&json!("Dune")));
    }

    #[test]
    fn simd_scan_descendant_matches_tree_walker() {
        use crate::Jetro;
        let raw = br#"{"a":{"test":1},"b":[{"test":2},{"other":9},{"test":3}],"comment":"the \"test\": lie"}"#.to_vec();
        let j_bytes = Jetro::from_bytes(raw.clone()).unwrap();
        let j_tree  = Jetro::new(serde_json::from_slice(&raw).unwrap());
        assert_eq!(j_bytes.collect("$..test").unwrap(),
                   j_tree.collect("$..test").unwrap());
    }

    #[test]
    fn simd_scan_chains_further_steps() {
        use crate::Jetro;
        let raw = br#"{"users":[{"id":1,"name":"a"},{"id":2,"name":"b"},{"id":3,"name":"c"}]}"#.to_vec();
        let j = Jetro::from_bytes(raw).unwrap();
        let r = j.collect("$..id.sum()").unwrap();
        assert_eq!(r, json!(6));
    }
}
