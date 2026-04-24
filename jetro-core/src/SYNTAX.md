# Jetro Syntax Reference

Complete reference for the Jetro v2 expression language. Grammar source: `src/grammar.pest`. Builtin catalog: `src/eval/builtins.rs`.

Entry points:

| Call | Notes |
|------|-------|
| `jetro::query(expr, &doc)` | One-shot, tree-walker |
| `jetro::query_with(expr, &doc, registry)` | With custom methods |
| `Jetro::new(doc).collect(expr)` | Thread-local cached VM |
| `Engine::new().run(expr, &doc)` | Shared-cache VM (multi-thread) |
| `jetro!("expr")` | Compile-time checked `Expr<Value>` (feature `macros`) |

---

## 1. Root and Context

| Token | Meaning |
|-------|---------|
| `$`   | Document root |
| `@`   | Current item (inside lambdas, pipes, comprehensions, patch values, map-into-shape bodies) |

---

## 2. Literals

```
null  true  false
42    3.14
"hello"  'world'             // double or single quotes
f"Hello {$.user.name}!"     // f-string: {expr}
f"{$.price:.2f}"            // format spec after `:`
f"{$.name | upper}"         // piped transform inside placeholder
```

F-strings capture raw between `f"` and `"`. Escape `"` with `\"`. Inner `{expr}` is parsed as a Jetro expression.

---

## 3. Navigation

### Field access

```
$.field
$.a.b.c
```

Field names allow `A-Z a-z 0-9 _ -` (first char alpha/`_`).

### Optional (null-safe) field

The `?` marker is **postfix** — it attaches to the step it guards, not
the next step. Prefix `?.field` is not accepted.

```
$.user?.name              // null if .user missing; .name runs on result
$.orders[0].total?        // total evaluated; null-safe at the end
$.a?.b?.c?                // chained null-safe field access
```

### Optional on descendant / method — null-safety only

Postfix `?` is **null-safety**, never first-of-array. It guards the
chain so that subsequent steps do not blow up on a null receiver.
To take the first element of an array use `.first()` explicitly.

```
$..services               // array of all `services` descendants
$..services?              // same array, null-safe
$..services?.first()      // first matched services obj (or null)

$.books.filter(price > 10)         // filtered array
$.books.filter(price > 10).first() // first book with price > 10
```

### `!` — exactly-one quantifier

`!` keeps its meaning: expect exactly one element, error otherwise.

```
$.books{title == "1984"}!          // error if 0 or >1 matches
```

### Array index

```
$.items[0]                // first
$.items[-1]               // last
$.items[n]                // dynamic (variable)
$.items[($.x).to_string()] // expression index
```

### Slice

```
$.items[2:5]              // half-open [2,5)
$.items[2:]               // from 2 to end
$.items[:5]               // first 5
```

### Recursive descent

```
$..title                  // all "title" anywhere
$..                       // all values (every node)
```

### Dynamic field

```
$.obj.{$.key_name}        // look up field named by $.key_name
```

### Inline filter (predicate postfix)

```
$.items{price > 10}       // filter items by predicate
```

### Quantifier

```
$.items?                  // reduce: present/null — returns null if empty
$.items!                  // assert non-empty; error if empty
```

---

## 4. Operators

### Comparison

```
== != < <= > >=
~=                        // fuzzy: case-insensitive substring
```

### Arithmetic

```
+ - * / %
-x                        // unary negate
```

### Logical

```
and  or  not
```

### Null-coalescing

```
$.field ?| "default"      // first non-null
$.a ?? $.b ?? "fallback"  // chained
```

`??` and `?|` are equivalent aliases.

### Cast

```
$.val as int
$.val as float
$.val as string
$.val as bool
$.s   as number
```

Cast types: `int`, `float`, `number`, `string`, `bool`, `array`, `object`, `null`.

### Precedence (low → high)

```
|  |>          (pipeline)
?? ?|          (coalesce)
or
and
not
kind / is
== != < <= > >= ~=
+ -
* / %
as
unary -
postfix (., [], (), ?, ?:)
```

---

## 5. Kind Checks

```
$.items.filter(v kind number)
$.items.filter(v is string)         // `is` is an alias for `kind`
$.items.filter(v kind not null)
$.events.filter(user_id is not array)
```

Kind types: `number`, `string`, `bool`, `array`, `object`, `null`.

---

## 6. Filter / Map / Chaining

```
$.books.filter(price > 10 and rating >= 4)
$.books.filter(lambda b: b.tags.includes("sci-fi"))
$.books.filter(title ~= "dune")
$.users.map(name)
$.users.map({name, role})
$.products.map({id, title: name, cost: price, stock: stock > 0})
$.nums.map(lambda n: n * n)
$.books.filter(price > 10).sort(-price).map({title, price}).count()
```

---

## 7. Aggregates

| Method | Notes |
|--------|-------|
| `count()` | Array length |
| `count(pred)` | Count matching predicate |
| `sum(field_or_lambda?)` | Sum numeric / field |
| `min(field?)` / `max(field?)` | Min/max value |
| `avg(field?)` | Arithmetic mean |
| `any(pred)` / `all(pred)` | Existential / universal |
| `group_by(key)` | `{key: [items]}` |
| `index_by(key)` | `{key: item}` |
| `count_by(key)` | `{key: N}` |
| `len()` | Length (array / string / object) |

---

## 8. Lambda

Classic form:

```
lambda n: n * n
lambda x, y: x + y
lambda p: p.price > 10 and p.stock > 0
```

Arrow form (same semantics):

```
n => n * n
(x, y) => x + y
() => 42
```

---

## 9. Let Bindings

Single:

```
let top = $.books.filter(price > 100)
in {count: top.len(), titles: top.map(title)}
```

Multi (nested desugar):

```
let users_idx = $.users.index_by(id),
    orders    = $.orders.filter(active)
in orders.map({id, user: users_idx[(user_id).to_string()].name})
```

---

## 10. Pipeline

`|` (or `|>`) forwards left value as the new context `@`.

```
$.products | filter(price < 20) | map(name) | sort
$.value    | to_string()
$.nums     | sum()
$.s        | .trim().upper()
```

Right-hand forms:
- `ident(args)` — method call on piped value
- `ident` — zero-arg method (or naked identifier as @ field)
- any expression — evaluated with `@` bound to piped value

---

## 11. Bind (`->`)

Capture the pipeline value without consuming it.

```
$.users -> users | users.filter(active).count()
```

Destructure object:

```
$.point -> {x, y} | x + y
$.user  -> {name, role, ...rest} | {name, extra: rest}
```

Destructure array:

```
$.pair -> [a, b] | a * b
```

---

## 12. Comprehensions

### List

```
[book.title for book in $.books if book.price > 10]
[x * 2 for x in $.numbers]
```

### Dict

```
{u.id: u.name for u in $.users if u.active}
```

### Set (unique)

```
{book.genre for book in $.books}
```

### Generator

```
(x for x in $.items if x > 0)
```

### Two-variable (key, value)

```
[k for k, v in $.obj]
{k: v * 2 for k, v in $.obj}
```

---

## 13. Object Construction

Seven field forms, any combination:

```
{
  name,                          // shorthand: name: name
  title: book.title,             // keyed
  cost: price * 1.2,             // computed
  slug?: $.maybe_slug,           // optional (drop if value is null)
  active?,                       // optional shorthand (drop if null)
  [dyn]: value,                  // dynamic key
  ...base,                       // shallow spread
  ...**defaults,                 // deep (recursive) spread
}
```

Conditional include — `when` guard on any keyed field:

```
{name, email: $.email when $.verified}
{grade: "pass" when score > threshold}
```

---

## 14. Array Construction

```
[1, 2, 3]
[...arr1, ...arr2]              // spread
[first, ...rest, last]          // mixed
```

---

## 15. Map-Into-Shape `[* if …] => expr`

Combines filter + map into a single postfix template.

```
$.store.books[*] => {title, price}
$.store.books[* if price > 10] => {title}
$.groups[*] => {name, ns: items[*] => n}
```

Equivalent to:

```
$.store.books.filter(...).map(...)
```

Use it when projection is the whole point of the pipeline.

---

## 16. Patch Blocks

`patch TARGET { key.path: value, ... }` — builds a new value with deep-path mutations. Original is untouched.

```
patch $ { name: "Bob" }                  // overwrite top field
patch $ { user.name: "Bob" }             // deep path
patch $ { age: 42 }                      // add new field
patch $ { tmp: DELETE }                  // delete key
```

### Path steps

| Step | Meaning |
|------|---------|
| `.field` | Descend by key |
| `[n]`    | Descend by index |
| `[*]`    | Wildcard (all elements) |
| `[* if pred]` | Filtered wildcard |
| `..field`| Recursive descent by key |

```
patch $ { users[*].seen: true }
patch $ { users[* if active].role: "admin" }
patch $ { users[*].email: @.lower() }          // @ = current value at path
patch $ { items[1]: 99 }
patch $ { users[* if not active]: DELETE }     // bulk delete
patch $ { ..name: @.upper() }                  // all "name" anywhere
```

### Conditional (`when`)

```
patch $ { count: @ + 1 when $.enabled }
```

If the guard is false/null, the field is left unchanged.

**Context note:** `when` is evaluated against the root document, not the matched element. To filter per-element (e.g. "every book where `stock == 0`"), put the predicate in the path with `[* if pred]`, whose context *is* the element:

```
patch $ {
    store.books[* if stock == 0].available: false,   // per-element
    tombstone: true when $.meta.deleted             // root-scoped
}
```

### Composition

`patch` is an expression — composes anywhere:

```
patch $ { name: "Bob" } | @.name
patch $ { name: "Bob" }.keys()
{result: patch $ { name: "Bob" }}
let x = patch $ { name: "Bob" } in x.name
patch (patch $ { name: "Bob" }) { age: 99 }
$.users.map(patch @ { n: @ * 10 })
```

`DELETE` is a sentinel — only valid as a patch-field value. Using it elsewhere is a runtime error.

---

## 17. Global Functions

```
coalesce(a, b, c)        // first non-null
chain(arr1, arr2)        // concatenate arrays
zip(arr1, arr2)          // [[a0,b0], [a1,b1], ...]
zip_longest(arr1, arr2)  // pad with null
product(arr1, arr2)      // cartesian product
to_string(v)             // free-function form
type_of(v)               // "number" | "string" | ...
```

Any builtin method can be called as a free function with the receiver as the first argument:

```
to_string($.user.id)      // same as $.user.id.to_string()
len($.items)              // same as $.items.len()
```

---

## 18. Null Safety Cheat-Sheet

```
$.user?.name                 // postfix ? on .user — null if user missing
$.users[0].profile?.bio      // ? on .profile — null-safe field
$.items?.first()             // array passthrough + explicit first
$.field ?| "default"         // coalesce
coalesce($.a, $.b, "x")      // first non-null
has(obj, "key")              // existence check
missing(obj, "key")          // negated existence
```

---

## 19. `jetro!` Macro (`features = ["macros"]`)

Compile-time checked expression literal.

```rust
use jetro::prelude::*;
use jetro::jetro;

let e = jetro!("$.books.filter(price > 10).map(title)");
let out = e.eval(&doc)?;
```

Checks at compile time:
- Balanced `()`, `[]`, `{}`
- Balanced `"` / `'` (escape-aware)
- Non-empty body

Full parse runs at first eval.

---

## 20. `#[derive(JetroSchema)]` (`features = ["macros"]`)

Attach a fixed set of named expressions to a type.

```rust
use jetro::prelude::*;
use jetro::JetroSchema;

#[derive(JetroSchema)]
#[expr(titles = "$.books.map(title)")]
#[expr(count  = "$.books.len()")]
#[expr(top    = "$.books.filter(price > 10)")]
struct BookView;

for (name, src) in BookView::exprs() {
    session.register_expr(name, src)?;
}
```

Generated:
```rust
impl JetroSchema for BookView {
    const EXPRS: &'static [(&'static str, &'static str)] = &[...];
    fn exprs() -> &'static [(&'static str, &'static str)] { Self::EXPRS }
    fn names() -> &'static [&'static str] { ... }
}
```

Each `#[expr(name = "src")]` is lex-checked at compile time (same rules as `jetro!`).

---

## 21. Method Catalog

Every snake_case method has a camelCase alias (e.g. `group_by` ≡ `groupBy`). Listed once here.

### Core

| Method | Purpose |
|--------|---------|
| `len` | Length (arr / str / obj) |
| `type` | Type name |
| `to_string` | Stringify |
| `to_json` / `from_json` | JSON (de)serialise |
| `has(key, ...)` | All keys present |
| `missing(key)` | Key absent |
| `or(default)` | Replace null |
| `set(key, val)` / `update(key, lambda)` | Return new object |
| `includes(v)` / `contains(v)` | Contains check |

### Objects

```
keys  values  entries  to_pairs  from_pairs  invert
pick(keys...)  omit(keys...)
merge(o)  deep_merge(o)  defaults(o)  rename(map)
transform_keys(fn)  transform_values(fn)
filter_keys(fn)  filter_values(fn)
pivot(row_key, col_key, val_key)
```

### Arrays

```
filter(pred)  map(expr)  flat_map(expr)
sort(key?)          // +field asc, -field desc, or lambda(a,b)→bool
flatten(n?)         // one level by default
reverse  unique (distinct)  compact  pairwise  enumerate
first  last  nth(n)  take(n)  drop(n)
take_while(pred)  drop_while(pred)
append(v)  prepend(v)  remove(v_or_pred)
includes(v)  index_of(v)  last_index_of(v)
diff(other)  intersect(other)  union(other)
window(n)  chunk(n)  batch(n)  accumulate(fn, init?)
partition(pred)      // {pass, fail}
zip(other)  zip_longest(other)
equi_join(other, left_key, right_key)
join(sep)            // array → string
slice(start, end?)
```

### Aggregates

```
sum(field?)  min(field?)  max(field?)  avg(field?)
count(pred?)  any(pred)  all(pred)
group_by(key)  index_by(key)  count_by(key)
```

### Strings

```
upper  lower  capitalize  title_case
trim  trim_left  trim_right
lines  words  chars
starts_with(s)  ends_with(s)
strip_prefix(s)  strip_suffix(s)
replace(old, new)  replace_all(old, new)
index_of(s)  last_index_of(s)
split(sep)  join(sep)   // (join is also an array method)
indent(n, pad?)  dedent
pad_left(n, ch?)  pad_right(n, ch?)
repeat(n)  slice(start, end?)
to_number  to_bool
to_base64 / from_base64
url_encode / url_decode
html_escape / html_unescape
matches(regex)       // bool
scan(regex)          // array of matches
```

### Paths

```
get_path("a.b.c")           // deep get
set_path("a.b.d", 42)       // deep set (new value)
del_path("a.b.d")           // deep delete
del_paths(["a.b", "c"])     // bulk delete
has_path("a.b.c")           // bool
flatten_keys()              // {"a.b.c": v}
unflatten_keys()            // reconstruct nested
```

### CSV / TSV

```
to_csv    // array of objects → CSV string (header = union of keys)
to_tsv    // TAB-separated variant
```

---

## 22. Reserved Keywords

```
true  false  null
and  or  not
for  in  if
let  lambda
kind  is  as  when
patch  DELETE
```

Cannot be used as identifiers. `DELETE` must be uppercase and only appears as a patch-field value.

---

## 23. Full Query Examples

```
// Books > $10, sorted desc, titles only
$.store.books.filter(price > 10).sort(-price).map(title)

// Projection-heavy: same via map-into-shape
$.store.books[* if price > 10] => {title, price}

// Active users with high scores
$.users.filter(active and score >= 90).map(name)

// Join via index
let users_idx = $.users.index_by(id)
in $.orders.map({id, total, user: users_idx[(user_id).to_string()].name})

// Equi-join built-in
$.orders.equi_join($.users, "user_id", "id").map({
  id, total, name: right.name
})

// Pivot
$.sales.pivot("region", "product", "amount")

// Recursive: all titles anywhere
$..title

// Comprehension with filter
[u.name for u in $.users if u.active and u.score > 70]

// Dict comprehension
{u.id: u.name for u in $.users if u.active}

// Pipeline
$.products | filter(price < 20) | map(name) | sort

// F-string with format
$.users.map(f"Hello {name}, your score is {score:.1f}")

// Patch: flag users whose last seen is old
patch $ { users[* if last_seen < threshold].status: "stale" }

// Patch: delete soft-deleted rows
patch $ { rows[* if deleted_at kind not null]: DELETE }

// Layered patch
let base = patch $ { meta.updated_at: now }
in patch base { rows[*].version: @.version + 1 }

// Conditional field (`when`)
{name, email: $.email when $.verified}

// Group then aggregate
$.orders.group_by(region).transform_values(lambda v: v.sum(total))

// Cross join via comprehension
[{u: u.name, p: p.name} for u in $.users for p in $.products if p.owner == u.id]
```
