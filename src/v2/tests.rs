#[cfg(test)]
mod tests {
    use serde_json::json;
    use crate::v2::query;

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
    fn optional_field_omitted() {
        let doc = json!({"user": {"name": "Alice"}});
        let r = query("$.user.map({name, email?})", &doc);
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
        let doc = json!({"a": {"x": 1}, "b": {"y": 2}});
        let r = query("$.a.merge($.b)", &doc).unwrap();
        assert_eq!(r["x"], json!(1));
        assert_eq!(r["y"], json!(2));
    }

    #[test]
    fn deep_merge_objects() {
        let doc = json!({"a": {"x": {"p": 1}}, "b": {"x": {"q": 2}, "y": 3}});
        let r = query("$.a.deep_merge($.b)", &doc).unwrap();
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
    fn fusion_find_first_opcode_emitted() {
        use crate::v2::vm::{Compiler, Opcode};
        let prog = Compiler::compile_str("$.books{price > 10}?").unwrap();
        let has_find_first = prog.ops.iter().any(|o| matches!(o, Opcode::FindFirst(_)));
        assert!(has_find_first, "FindFirst opcode not emitted; got: {:?}", prog.ops);
    }

    #[test]
    fn fusion_find_one_opcode_emitted() {
        use crate::v2::vm::{Compiler, Opcode};
        let prog = Compiler::compile_str("$.books{title == \"x\"}!").unwrap();
        let has_find_one = prog.ops.iter().any(|o| matches!(o, Opcode::FindOne(_)));
        assert!(has_find_one, "FindOne opcode not emitted; got: {:?}", prog.ops);
    }

    #[test]
    fn fusion_find_first_from_filter_method() {
        use crate::v2::vm::{Compiler, Opcode};
        let prog = Compiler::compile_str("$.books.filter(price > 10)?").unwrap();
        let has_find_first = prog.ops.iter().any(|o| matches!(o, Opcode::FindFirst(_)));
        assert!(has_find_first, "FindFirst should fuse from .filter() too");
    }

    #[test]
    fn redundant_reverse_eliminated() {
        use crate::v2::vm::{Compiler, Opcode, BuiltinMethod};
        let prog = Compiler::compile_str("$.books.reverse().reverse()").unwrap();
        let reverse_count = prog.ops.iter().filter(|o| matches!(o,
            Opcode::CallMethod(c) if c.method == BuiltinMethod::Reverse
        )).count();
        assert_eq!(reverse_count, 0, "adjacent reverse+reverse should cancel");
    }

    #[test]
    fn redundant_unique_collapsed() {
        use crate::v2::vm::{Compiler, Opcode, BuiltinMethod};
        let prog = Compiler::compile_str("$.books.unique().unique()").unwrap();
        let unique_count = prog.ops.iter().filter(|o| matches!(o,
            Opcode::CallMethod(c) if c.method == BuiltinMethod::Unique
        )).count();
        assert_eq!(unique_count, 1, "duplicate unique() should collapse");
    }

    #[test]
    fn bool_short_circuit_folded() {
        use crate::v2::vm::{Compiler, Opcode};
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
        use crate::v2::vm::{Compiler, Opcode};
        let prog = Compiler::compile_str("42 kind number").unwrap();
        let has_kind = prog.ops.iter().any(|o| matches!(o, Opcode::KindCheck { .. }));
        assert!(!has_kind, "literal kind check should fold away");
        assert!(matches!(prog.ops.first(), Some(Opcode::PushBool(true))));
    }

    #[test]
    fn kind_check_literal_fold_mismatch() {
        use crate::v2::vm::{Compiler, Opcode};
        let prog = Compiler::compile_str("\"hi\" kind number").unwrap();
        assert!(matches!(prog.ops.first(), Some(Opcode::PushBool(false))));
        assert!(!prog.ops.iter().any(|o| matches!(o, Opcode::KindCheck { .. })));
    }

    #[test]
    fn kind_check_literal_negate() {
        use crate::v2::vm::{Compiler, Opcode};
        let prog = Compiler::compile_str("42 kind not string").unwrap();
        assert!(matches!(prog.ops.first(), Some(Opcode::PushBool(true))));
        assert!(!prog.ops.iter().any(|o| matches!(o, Opcode::KindCheck { .. })));
    }

    #[test]
    fn analysis_infer_result_type() {
        use crate::v2::analysis::{infer_result_type, VType};
        use crate::v2::vm::Compiler;
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
        use crate::v2::analysis::count_ident_uses;
        use crate::v2::vm::Compiler;
        let p = Compiler::compile_str("let x = 10 in x + x + 1").unwrap();
        assert_eq!(count_ident_uses(&p, "x"), 2);
        let p = Compiler::compile_str("let y = 10 in 42").unwrap();
        assert_eq!(count_ident_uses(&p, "y"), 0);
    }

    #[test]
    fn analysis_collect_accessed_fields() {
        use crate::v2::analysis::collect_accessed_fields;
        use crate::v2::vm::Compiler;
        let p = Compiler::compile_str("$.store.books.map(@.title)").unwrap();
        let fields = collect_accessed_fields(&p);
        assert!(fields.iter().any(|f| f.as_ref() == "store"));
        assert!(fields.iter().any(|f| f.as_ref() == "books"));
        assert!(fields.iter().any(|f| f.as_ref() == "title"));
    }

    #[test]
    fn analysis_program_signature_stable() {
        use crate::v2::analysis::program_signature;
        use crate::v2::vm::Compiler;
        let a = Compiler::compile_str("$.x.y + 1").unwrap();
        let b = Compiler::compile_str("$.x.y + 1").unwrap();
        assert_eq!(program_signature(&a), program_signature(&b));
        let c = Compiler::compile_str("$.x.y + 2").unwrap();
        assert_ne!(program_signature(&a), program_signature(&c));
    }

    #[test]
    fn dead_let_eliminated() {
        use crate::v2::vm::{Compiler, Opcode};
        let prog = Compiler::compile_str("let x = 100 in 42").unwrap();
        let has_let = prog.ops.iter().any(|o| matches!(o, Opcode::LetExpr { .. }));
        assert!(!has_let, "unused let should be eliminated");
        assert!(matches!(prog.ops.as_ref(), [Opcode::PushInt(42)]));
    }

    #[test]
    fn dead_let_kept_when_used() {
        use crate::v2::vm::{Compiler, Opcode};
        let prog = Compiler::compile_str("let x = 10 in x + 1").unwrap();
        assert!(prog.ops.iter().any(|o| matches!(o, Opcode::LetExpr { .. })));
    }

    #[test]
    fn const_fold_comparison() {
        use crate::v2::vm::{Compiler, Opcode};
        let prog = Compiler::compile_str("1 < 2").unwrap();
        assert!(matches!(prog.ops.as_ref(), [Opcode::PushBool(true)]));
        let prog = Compiler::compile_str("5 == 5").unwrap();
        assert!(matches!(prog.ops.as_ref(), [Opcode::PushBool(true)]));
        let prog = Compiler::compile_str("\"a\" == \"b\"").unwrap();
        assert!(matches!(prog.ops.as_ref(), [Opcode::PushBool(false)]));
    }

    #[test]
    fn const_fold_unary() {
        use crate::v2::vm::{Compiler, Opcode};
        let prog = Compiler::compile_str("not true").unwrap();
        assert!(matches!(prog.ops.as_ref(), [Opcode::PushBool(false)]));
        let prog = Compiler::compile_str("-42").unwrap();
        assert!(matches!(prog.ops.as_ref(), [Opcode::PushInt(-42)]));
    }

    #[test]
    fn fusion_topn_opcode() {
        use crate::v2::vm::{Compiler, Opcode};
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
        use crate::v2::analysis::{infer_monotonicity, Monotonicity};
        use crate::v2::vm::Compiler;
        let p = Compiler::compile_str("$.x.sort()").unwrap();
        assert_eq!(infer_monotonicity(&p), Monotonicity::Asc);
        let p = Compiler::compile_str("$.x.sort().reverse()").unwrap();
        assert_eq!(infer_monotonicity(&p), Monotonicity::Desc);
    }

    #[test]
    fn analysis_cost_nonzero() {
        use crate::v2::analysis::program_cost;
        use crate::v2::vm::Compiler;
        let cheap = Compiler::compile_str("42").unwrap();
        let expensive = Compiler::compile_str("$.books.filter(@.price > 10).map(@.title).sort()").unwrap();
        assert!(program_cost(&cheap) < program_cost(&expensive));
    }

    #[test]
    fn analysis_selectivity_score() {
        use crate::v2::analysis::selectivity_score;
        use crate::v2::parser::parse;
        let eq = parse("x == 1").unwrap();
        let lt = parse("x < 1").unwrap();
        let t  = parse("true").unwrap();
        assert!(selectivity_score(&eq) < selectivity_score(&lt));
        assert!(selectivity_score(&lt) < selectivity_score(&t));
    }

    #[test]
    fn analysis_escapes_doc() {
        use crate::v2::analysis::escapes_doc;
        use crate::v2::vm::Compiler;
        let p = Compiler::compile_str("42").unwrap();
        assert!(!escapes_doc(&p));
        let p = Compiler::compile_str("$.x").unwrap();
        assert!(escapes_doc(&p));
    }

    #[test]
    fn fusion_map_sum_opcode() {
        use crate::v2::vm::{Compiler, Opcode};
        let prog = Compiler::compile_str("$.books.map(@.price).sum()").unwrap();
        let has = prog.ops.iter().any(|o| matches!(o, Opcode::MapSum(_)));
        assert!(has, "map+sum should fuse to MapSum");
    }

    #[test]
    fn fusion_map_avg_opcode() {
        use crate::v2::vm::{Compiler, Opcode};
        let prog = Compiler::compile_str("$.books.map(@.price).avg()").unwrap();
        let has = prog.ops.iter().any(|o| matches!(o, Opcode::MapAvg(_)));
        assert!(has, "map+avg should fuse to MapAvg");
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
        use crate::v2::analysis::dedup_subprograms;
        use crate::v2::vm::Compiler;
        // Two identical sub-exprs inside an array — after dedup they share Arc.
        let prog = Compiler::compile_str("[$.a.b + 1, $.a.b + 1]").unwrap();
        let deduped = dedup_subprograms(&prog);
        // Extract MakeArr sub-progs and confirm Arc identity.
        use crate::v2::vm::Opcode;
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
        use crate::v2::analysis::find_common_subexprs;
        use crate::v2::vm::Compiler;
        let prog = Compiler::compile_str("[$.x.y, $.x.y, $.z]").unwrap();
        let cs = find_common_subexprs(&prog);
        assert!(!cs.is_empty(), "should find at least one repeated sub-program");
    }

    #[test]
    fn method_const_fold_str_len() {
        use crate::v2::vm::{Compiler, Opcode};
        let prog = Compiler::compile_str("\"hello\".len()").unwrap();
        assert!(matches!(prog.ops.as_ref(), [Opcode::PushInt(5)]));
    }

    #[test]
    fn method_const_fold_str_upper() {
        use crate::v2::vm::{Compiler, Opcode};
        let prog = Compiler::compile_str("\"hi\".upper()").unwrap();
        assert!(matches!(prog.ops.as_ref(), [Opcode::PushStr(s)] if s.as_ref() == "HI"));
    }

    #[test]
    fn method_const_fold_arr_len() {
        use crate::v2::vm::{Compiler, Opcode};
        let prog = Compiler::compile_str("[1,2,3,4].len()").unwrap();
        assert!(matches!(prog.ops.as_ref(), [Opcode::PushInt(4)]));
    }

    #[test]
    fn analysis_expr_uses_ident() {
        use crate::v2::analysis::expr_uses_ident;
        use crate::v2::parser::parse;
        let e = parse("x + 1").unwrap();
        assert!(expr_uses_ident(&e, "x"));
        assert!(!expr_uses_ident(&e, "y"));
        let e = parse("let x = 10 in x + 1").unwrap();
        assert!(!expr_uses_ident(&e, "x"), "inner let shadows");
    }

    #[test]
    fn analysis_fold_kind_check_helper() {
        use crate::v2::analysis::{fold_kind_check, VType};
        use crate::v2::ast::KindType;
        assert_eq!(fold_kind_check(VType::Int, KindType::Number, false), Some(true));
        assert_eq!(fold_kind_check(VType::Str, KindType::Number, false), Some(false));
        assert_eq!(fold_kind_check(VType::Str, KindType::Str, true),    Some(false));
        assert_eq!(fold_kind_check(VType::Unknown, KindType::Str, false), None);
    }
}
