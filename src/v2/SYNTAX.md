# Jetro v2 Syntax Reference

## Root and Context

| Token | Meaning |
|-------|---------|
| `$`   | Document root |
| `@`   | Current item (inside lambdas, pipes, comprehensions) |

---

## Navigation

### Field access
```
$.field
$.a.b.c
```

### Optional field (null-safe)
```
$.user?.name          // null if user is null, not an error
```

### Array index
```
$.items[0]            // first element
$.items[-1]           // last element
$.items[n]            // dynamic index (variable or expression)
$.items[(x).to_string()]  // expression index
```

### Slice
```
$.items[2:5]          // elements [2, 5) (exclusive end)
$.items[2:]           // from index 2 to end
$.items[:5]           // first 5 elements
```

### Recursive descent
```
$..title              // all "title" values anywhere in document
```

---

## Literals

```
null  true  false
42    3.14
"hello"  'world'
f"Hello {$.user.name}!"     // f-string with interpolation
f"{$.price:.2f}"            // f-string with format spec
f"{$.name | upper}"         // f-string with pipe transform
```

---

## Operators

### Comparison
```
==  !=  <  <=  >  >=
~=                          // fuzzy match (case-insensitive substring)
```

### Arithmetic
```
+  -  *  /  %
```

### Logical
```
and  or  not
```

### Null coalescing
```
$.field ?| "default"
```

---

## Kind checks

```
$.items.filter(v kind number)
$.items.filter(v kind string)
$.items.filter(v kind bool)
$.items.filter(v kind array)
$.items.filter(v kind object)
$.items.filter(v kind null)
$.items.filter(v kind not null)   // negated kind check
```

---

## Filter

```
$.books.filter(price > 10)
$.books.filter(price > 10 and rating >= 4)
$.books.filter(price < 20 or stock == 0)
$.books.filter(not active)
$.books.filter(title == "Dune")
$.books.filter(active == true)
$.books.filter(name ~= "widget")              // fuzzy/substring
$.books.filter(tags.includes("sci-fi"))       // method in predicate
$.books.filter(lambda b: b.price > 10)        // lambda predicate
$.events.filter(user_id kind not null)        // kind check predicate
```

---

## Map

```
$.users.map(name)                             // pluck field
$.users.map({name, role})                     // object shorthand
$.products.map({id, title: name, cost: price}) // rename fields
$.products.map({name, in_stock: stock > 0})   // computed fields
$.numbers.map(lambda n: n * n)                // lambda transform
$.products.map({name, color: meta.color})     // nested field access
```

---

## Chaining

```
$.books.filter(price > 10).map(title)
$.books.filter(price > 10).sort(-price).map({title, price})
$.books.filter(price > 10).count()
```

---

## Aggregates

| Method | Description |
|--------|-------------|
| `.count()` | Array length |
| `.sum(field)` | Sum of field values |
| `.min(field)` | Minimum field value |
| `.max(field)` | Maximum field value |
| `.avg(field)` | Average of field values |
| `.len()` | Length (array or string) |

```
$.books.filter(price > 10).count()
$.books.sum(price)
$.books.max(rating)
```

---

## Array methods

| Method | Description |
|--------|-------------|
| `.sort()` | Natural ascending sort |
| `.sort(field)` | Sort by field ascending |
| `.sort(-field)` | Sort by field descending |
| `.sort(lambda a, b: a.price < b.price)` | Custom comparator |
| `.reverse()` | Reverse array |
| `.unique()` | Remove duplicates |
| `.flatten()` | Flatten one level |
| `.flatten(n)` | Flatten n levels |
| `.flat_map(expr)` | Map then flatten one level |
| `.first()` | First element |
| `.last()` | Last element |
| `.take(n)` | First n elements |
| `.drop(n)` | Skip first n elements |
| `.take_while(pred)` | Take while predicate holds |
| `.drop_while(pred)` | Drop while predicate holds |
| `.nth(n)` | n-th element (0-based) |
| `.append(val)` | Add to end |
| `.prepend(val)` | Add to front |
| `.remove(val)` | Remove elements equal to val |
| `.remove(lambda v: pred)` | Remove elements matching predicate |
| `.includes(val)` | Array contains val |
| `.index_of(val)` | Index of first occurrence |
| `.join(sep)` | Join strings with separator |
| `.split(sep)` | Split string into array |
| `.group_by(expr)` | Group into `{key: [items]}` |
| `.index_by(expr)` | Index into `{key: item}` |
| `.counts_by(expr)` | Count by key |
| `.partition(pred)` | `{pass: [...], fail: [...]}` |
| `.zip(other)` | Zip two arrays |
| `.enumerate()` | `[{index, value}, ...]` |
| `.pairwise()` | Sliding pairs |
| `.window(n)` | Sliding windows of size n |
| `.diff(other)` | Set difference |
| `.intersect(other)` | Set intersection |
| `.union(other)` | Set union |
| `.compact()` | Remove nulls |
| `.sum()` | Sum numeric elements |
| `.min()` | Min element |
| `.max()` | Max element |
| `.avg()` | Average of elements |

---

## Object methods

| Method | Description |
|--------|-------------|
| `.keys()` | Array of keys |
| `.values()` | Array of values |
| `.entries()` | Array of `[key, value]` pairs |
| `.pick("k1", "k2")` | Keep only named keys |
| `.omit("k1", "k2")` | Remove named keys |
| `.merge(obj)` | Shallow merge (right wins) |
| `.deep_merge(obj)` | Deep recursive merge |
| `.defaults(obj)` | Fill null fields from obj |
| `.rename(map)` | Rename keys via `{old: "new"}` map |
| `.invert()` | Swap keys and values |
| `.transform_keys(lambda)` | Transform key names |
| `.transform_values(lambda)` | Transform values |
| `.filter_keys(lambda)` | Keep keys matching predicate |
| `.filter_values(lambda)` | Keep values matching predicate |
| `.to_pairs()` | `[{key, val}, ...]` |
| `.from_pairs()` | Reconstruct object from pairs |
| `.pivot("row", "col", "val")` | Nested pivot `{row: {col: val}}` |

---

## String methods

| Method | Description |
|--------|-------------|
| `.upper()` | Uppercase |
| `.lower()` | Lowercase |
| `.trim()` | Trim both ends |
| `.trim_left()` | Trim left |
| `.trim_right()` | Trim right |
| `.split(sep)` | Split to array |
| `.replace(old, new)` | Replace first occurrence |
| `.replace_all(old, new)` | Replace all occurrences |
| `.starts_with(s)` | Boolean check |
| `.ends_with(s)` | Boolean check |
| `.strip_prefix(s)` | Remove prefix |
| `.strip_suffix(s)` | Remove suffix |
| `.pad_left(n, char)` | Left-pad to width |
| `.pad_right(n, char)` | Right-pad to width |
| `.repeat(n)` | Repeat string n times |
| `.includes(s)` | Substring check |
| `.index_of(s)` | Index of substring |
| `.chars()` | Array of characters |
| `.to_string()` | Convert to string |
| `.to_int()` | Parse as integer |
| `.to_float()` | Parse as float |
| `.base64_encode()` | Base64 encode |
| `.base64_decode()` | Base64 decode |
| `.url_encode()` | URL encode |
| `.url_decode()` | URL decode |
| `.html_escape()` | Escape HTML entities |
| `.html_unescape()` | Unescape HTML entities |

---

## Path methods

```
$.nested.get_path("a.b.c")           // deep get by dot-path
$.nested.set_path("a.b.d", 999)      // deep set (returns new value)
$.nested.del_path("a.b.d")           // deep delete
$.nested.has_path("a.b.c")           // existence check
$.nested.flatten_keys()              // {a.b.c: val, ...}
$.flat.unflatten_keys()              // reconstruct nested from flat keys
```

---

## Lambda

```
lambda x: x * 2
lambda x, y: x + y
lambda p: p.price > 10 and p.stock > 0
```

Used inside methods:
```
$.numbers.map(lambda n: n * n)
$.users.filter(lambda u: u.score > 80 and u.active)
$.products.sort(lambda a, b: a.price < b.price)
$.vals.remove(lambda v: v % 2 == 0)
```

---

## Let bindings

```
let x = $.store.books in x.filter(price > 10).count()

let top = $.books.filter(price > 100) in {count: top.len(), titles: top.map(title)}

let users_idx = $.users.indexBy(id)
in $.orders.map({id, user: users_idx[(user_id).to_string()].name})
```

---

## Pipeline (`|`)

The `|` operator passes the left value as the context for the right expression.

```
$.products | filter(price < 20)
$.products | filter(price < 20) | map(name) | sort
$.value | .to_string()
$.numbers | .sum()
```

Right-hand side forms:
- `method_name(args)` — method call on piped value
- `ident` — method call with no args on piped value
- Any expression — evaluated with piped value as `@`

---

## Bind (`->`)

Captures the current pipeline value into a variable without consuming it:

```
$.users -> users | users.filter(active).count()
```

Destructuring:
```
$.point -> {x, y} | x + y
$.pair  -> [a, b] | a * b
```

---

## Comprehensions

### List comprehension
```
[book.title for book in $.books if book.price > 10]
[x * 2 for x in $.numbers.ints]
```

### Dict comprehension
```
{user.id: user.name for user in $.users if user.active}
```

### Set comprehension (unique values)
```
{book.genre for book in $.books}
```

---

## Object construction

```
{name, role}                          // shorthand: field: field
{title: name, cost: price}            // rename
{name, score: price * 2}              // computed
{...base, extra: "value"}             // spread
{[dynamic_key]: value}                // dynamic key
{optional_key?: value}                // omit if value is null
```

---

## Array construction

```
[1, 2, 3]
[...arr1, ...arr2]                    // spread
[a, b, ...rest]                       // mixed
```

---

## Global functions

```
coalesce(a, b, c)           // first non-null
chain(arr1, arr2)            // concatenate arrays
zip(arr1, arr2)              // zip to [[a1,b1], [a2,b2], ...]
zip_longest(arr1, arr2)      // zip with null padding
product(arr1, arr2)          // cartesian product
to_string(value)             // convert to string
type_of(value)               // type name: "number", "string", etc.
```

Any builtin method can also be called as a free function with the receiver as the first argument:
```
to_string($.user.id)         // same as $.user.id.to_string()
```

---

## Null safety

```
$.user?.name                 // null if user is null
$.users[0]?.profile?.bio     // chained null safety
$.field ?| "default"         // null coalescing
```

---

## Type conversion methods

```
$.value.to_string()
$.value.to_int()
$.value.to_float()
$.value.type_of()            // returns "number", "string", "bool", "array", "object", "null"
```

---

## Full query examples

```
// Books over $10, sorted by price descending, titles only
$.store.books.filter(price > 10).sort(-price).map(title)

// Active users with high scores
$.users.filter(active == true and score >= 90).map(name)

// Join-like: enrich orders with user names
let users_idx = $.users.indexBy(id)
in $.orders.map({id, total, user: users_idx[(user_id).to_string()].name})

// Pivot sales data
$.pivot_data.pivot("region", "product", "sales")

// Group products by category
$.products.group_by(category)

// Recursive: all titles anywhere in doc
$..title

// Comprehension with filter
[u.name for u in $.users if u.active and u.score > 70]

// Dict comprehension
{u.id: u.name for u in $.users if u.active}

// Pipeline
$.products | filter(price < 20) | map(name) | sort

// F-string
$.users.map(f"Hello {name}, your score is {score:.1f}")
```
