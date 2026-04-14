# jetro

[<img src="https://img.shields.io/badge/docs-jetro-blue"></img>](https://docs.rs/jetro)
[<img src="https://img.shields.io/badge/try-online%20repl-brightgreen"></img>](https://jetro.io)
![GitHub](https://img.shields.io/github/license/mitghi/jetro)

Jetro is a library which provides a custom DSL for transforming, querying and comparing data in JSON format. It is easy to use and extend.

Jetro has minimal dependency, the traversal and eval algorithm is implemented on top of [serde_json](https://serde.rs).

Jetro can be used inside Web Browser by compiling down to WASM. [Clone it](https://github.com/mitghi/jetroweb) and give it a shot.

Jetro can be used in command line using [Jetrocli](https://github.com/mitghi/jetrocli).

Jetro combines access paths with functions that operate on values matched within the pipeline. Access paths use `/` as separator (similar to URI structure). The start of a path denotes the traversal root:

- `>` — traverse from document root
- `<` — traverse from root in nested paths
- `@` — current pipeline item (used inside construction literals and `#map`)

Expressions support line breaks and whitespace; statements can be broken into multiple lines. Functions are denoted with the `#` operator and can be composed.

---

## Language Overview

### Path Navigation

| Syntax | Description |
| ------ | ----------- |
| `>/key` | Child field access |
| `>/*` | Any child (wildcard) |
| `>/..key` | Recursive descendant search |
| `>/..('k'='v')` | Descendant where field equals value |
| `>/('a' \| 'b')` | First matching key among alternatives |
| `>/[n]` | Array index access |
| `>/[n:m]` | Array slice |
| `>/[:n]` | Array from start up to index n |
| `>/[n:]` | Array from index n to end |

### Construction Literals

Jetro expressions can construct new JSON objects and arrays inline. Each field value is itself a Jetro expression.

**Object construction:**

```
>{ "key": expression, [dynamic_key_expr]: expression, ... }
```

Keys can be static string literals (single- or double-quoted) or dynamic expressions wrapped in `[...]`.

**Array construction:**

```
>[ expression, expression, ... ]
```

**`@` current-item path** refers to the current pipeline element and is usable inside construction literals and `#map`:

```
>/items/#map(>{ "label": @/name, "price_incl_tax": @/price })
```

---

## Functions

### Existing functions

| Function | Description |
| -------- | ----------- |
| `#pick('key' \| expr, ...)` | Select keys / sub-expressions to build a new object |
| `#head` | First element of an array |
| `#tail` | All elements except the first |
| `#last` | Last element of an array |
| `#nth(n)` | Element at index n |
| `#keys` | Keys of an object |
| `#values` | Values of an object |
| `#reverse` | Reverse an array |
| `#min` | Minimum numeric value |
| `#max` | Maximum numeric value |
| `#sum` | Sum of numeric values |
| `#len` | Length of array or object |
| `#all` | True when all boolean values are true |
| `#any` | True when at least one boolean value is true |
| `#not` | Logical negation of a boolean value |
| `#zip` | Zip two or more arrays into an array of objects |
| `#map(x: x.field)` | Map each element through a path expression |
| `#filter('field' op value [and\|or ...])` | Filter an array by condition |
| `#formats('{} {}', 'k1', 'k2') [-> \| ->* 'alias']` | Format a string from field values |

### Math functions

| Function | Description |
| -------- | ----------- |
| `#avg` | Average of numeric values |
| `#add(n)` | Add scalar n to each numeric value |
| `#sub(n)` | Subtract scalar n from each numeric value |
| `#mul(n)` | Multiply each numeric value by n |
| `#div(n)` | Divide each numeric value by n |
| `#abs` | Absolute value |
| `#round` | Round to nearest integer |
| `#floor` | Round down |
| `#ceil` | Round up |

### Array transforms

| Function | Description |
| -------- | ----------- |
| `#flatten` | Flatten one level of nested arrays |
| `#flat_map(x: x.field)` | Map then flatten |
| `#chunk(n)` | Split array into chunks of size n |
| `#unique` | Remove duplicate scalar values |
| `#distinct('key')` | Remove duplicate objects by a key field |
| `#sort_by('key' [, 'desc'])` | Sort objects by a field (ascending by default) |
| `#join_str('sep')` | Join string values with a separator |
| `#compact` | Remove null values from an array |
| `#count` | Count of elements (alias for `#len`) |

### Grouping & indexing

| Function | Description |
| -------- | ----------- |
| `#group_by('key')` | Group objects into `{ "value": [...] }` |
| `#count_by('key')` | Count objects per distinct key value |
| `#index_by('key')` | Index objects into `{ "value": object }` |
| `#tally` | Count occurrences of each scalar value |

### Object manipulation

| Function | Description |
| -------- | ----------- |
| `#merge` | Merge an array of objects into one |
| `#omit('key')` | Remove a key from an object |
| `#select('key')` | Keep only the given key in an object |
| `#rename('old', 'new')` | Rename a key |
| `#set('key', 'value')` | Set or overwrite a key with a static value |
| `#coalesce('default')` | Return default if the value is null |
| `#get('key')` | Extract a single field from an object in a pipeline |

### Join & lookup

| Function | Description |
| -------- | ----------- |
| `#join(>/other, 'left_key', 'right_key')` | Inner join: merge each left object with its match from right |
| `#lookup(>/other, 'left_key', 'right_key')` | Left join: merge left with first match from right (null-patch on miss) |

### Field resolution

These functions resolve references between collections — like foreign-key lookups in a query pipeline. They work both within a single document and across nodes in a `Graph`.

| Function | Description |
| -------- | ----------- |
| `#find(condition)` | First element in array matching a filter condition |
| `#filter_by(condition)` | All elements matching a filter condition |
| `#pluck('key')` | Extract one field from every object in an array |
| `#resolve('ref', >/target [, 'match'])` | Replace each item's reference field with the full matched object |
| `#deref(>/target [, 'match'])` | Current value IS the reference; return the matched object |

**`#find` / `#filter_by`** accept the same condition syntax as `#filter`:

```
>/items/#find('type' == 'ingredient')
>/items/#filter_by('price' > 2.0 and 'is_gratis' == false)
```

**`#resolve`** — for each object in the input array, look up the value of `ref_field` in the target collection and replace the scalar reference with the full matched object:

```
>/orders/#resolve('customer_id', >/customers, 'id')
```

**`#deref`** — the current value is itself the reference key; returns the first matched object:

```
>/order/customer_id/#deref(>/customers, 'id')
```

---

## Graph — multi-document queries

`Graph` lets you register named JSON documents and query across all of them. All nodes are merged into a virtual root `{ "node_name": value, ... }` that any Jetro expression can navigate.

```rust
use jetro::graph::Graph;

let mut g = Graph::new();
g.add_node("orders",    serde_json::json!([...]));
g.add_node("customers", serde_json::json!([...]));

// Query the virtual merged root
let result = g.query(">/orders/#resolve('customer_id', >/customers, 'id')")?;

// Query a specific node only
let result = g.query_node("customers", ">/customers/#filter('active' == true)")?;

// Build a message schema — values are expressions evaluated against the graph
let result = g.message(r#"
    {
      "total_revenue": ">/orders/..price/#sum",
      "customer_count": ">/customers/#len"
    }
"#)?;
```

---

## Quick-start example

```rust
let data = serde_json::json!({
  "name": "mr snuggle",
  "some_entry": {
    "some_obj": {
      "obj": {
        "a": "object_a",
        "b": "object_b",
        "c": "object_c",
        "d": "object_d"
      }
    }
  }
});

let mut values = Path::collect(data, ">/..obj/#pick('a','b')");

#[derive(Serialize, Deserialize)]
struct Output {
   a: String,
   b: String,
}

let output: Option<Output> = values.from_index(0);
```

---

## Example dataset

The following JSON is used in the query examples below.

```json
{
  "customer": {
    "id": "xyz",
    "ident": {
      "user": {
        "isExternal": false,
        "profile": {
          "firstname": "John",
          "alias": "Japp",
          "lastname": "Appleseed"
        }
      }
    },
    "preferences": []
  },
  "line_items": {
    "items": [
      { "ident": "abc", "is_gratis": false, "name": "pizza",       "price": 4.8, "total": 1,  "type": "base_composable" },
      { "ident": "def", "is_gratis": false, "name": "salami",      "price": 2.8, "total": 10, "type": "ingredient" },
      { "ident": "ghi", "is_gratis": false, "name": "cheese",      "price": 2,   "total": 1,  "type": "ingredient" },
      { "ident": "uip", "is_gratis": true,  "name": "chilli",      "price": 0,   "total": 1,  "type": "ingredient" },
      { "ident": "ewq", "is_gratis": true,  "name": "bread sticks","price": 0,   "total": 8,  "type": "box" }
    ]
  }
}
```

---

### Queries

Get value associated with `line_items`.

```
>/line_items
```

---

Get value associated with first matching key which has a value and return its `id` field.

```
>/('non-existing-member' | 'customer')/id
```

<details>
  <summary>See output</summary>

```json
"xyz"
```
</details>

---

Recursively search for objects that have a key with a specified value.

```
>/..('type'='ingredient')
```

<details>
  <summary>See output</summary>

```json
[
  { "ident": "ghi", "is_gratis": false, "name": "cheese",  "price": 2,   "total": 1,  "type": "ingredient" },
  { "ident": "def", "is_gratis": false, "name": "salami",  "price": 2.8, "total": 10, "type": "ingredient" }
]
```
</details>

---

Tail of the items list.

```
>/..items/#tail
```

---

Filter with compound condition.

```
>/..items/#filter('is_gratis' == true and 'name' ~= 'ChILLi')
```

<details>
  <summary>See output</summary>

```json
[
  { "ident": "uip", "is_gratis": true, "name": "chilli", "price": 0, "total": 1, "type": "ingredient" }
]
```
</details>

---

Filter then map.

```
>/..items/#filter('is_gratis' == true and 'name' ~= 'ChILLi')/#map(x: x.type)
```

<details>
  <summary>See output</summary>

```json
["ingredient"]
```
</details>

---

Construct a summary object.

```
>/#pick(
  >/..line_items/*/#filter('is_gratis' == false)/..price/#sum as 'total',
  >/..user/profile/#formats('{} {}', 'firstname', 'lastname') ->* 'fullname'
)
```

<details>
  <summary>See output</summary>

```json
{ "fullname": "John Appleseed", "total": 9.6 }
```
</details>

---

Slice the first four items.

```
>/..items/[:4]
```

---

Select from the fourth index to end.

```
>/..items/[4:]
```

---

Count gratis items.

```
>/#pick(>/..items/..is_gratis/#len as 'total_gratis')
```

<details>
  <summary>See output</summary>

```json
{ "total_gratis": 2 }
```
</details>

---

Keys and values of the first item.

```
>/..items/[0]/#keys
>/..items/[0]/#values
```

---

Zip two or more arrays together.

```
>/#pick(>/..name as 'name', >/..nested as 'field', >/..b as 'release')/#zip
```

<details>
  <summary>See output (JSON)</summary>

```json
{ "a": [{"name":"tool","value":{"nested":"field"}},{"name":"pneuma","value":{"nested":"seal"}}], "b": [2000,2100] }
```

Result:

```json
[
  { "field": "field", "name": "tool",   "release": 2000 },
  { "field": "seal",  "name": "pneuma", "release": 2100 }
]
```
</details>

---

Group items by type, then count.

```
>/..items/#group_by('type')
>/..items/#count_by('type')
```

<details>
  <summary>See output for count_by</summary>

```json
{ "base_composable": 1, "box": 1, "ingredient": 3 }
```
</details>

---

Sort items by price descending, then pluck their names.

```
>/..items/#sort_by('price', 'desc')/#pluck('name')
```

<details>
  <summary>See output</summary>

```json
["pizza", "salami", "cheese", "chilli", "bread sticks"]
```
</details>

---

Construct a new object for each item using `@` (current-item path) inside an object literal.

```
>/..items/#map(>{ "label": @/name, "unit_price": @/price, "gratis": @/is_gratis })
```

<details>
  <summary>See output</summary>

```json
[
  { "label": "pizza",       "unit_price": 4.8, "gratis": false },
  { "label": "salami",      "unit_price": 2.8, "gratis": false },
  { "label": "cheese",      "unit_price": 2,   "gratis": false },
  { "label": "chilli",      "unit_price": 0,   "gratis": true  },
  { "label": "bread sticks","unit_price": 0,   "gratis": true  }
]
```
</details>

---

Find the first ingredient and resolve it — example with a separate catalogue document.

```rust
let catalogue = serde_json::json!([
    { "ident": "abc", "description": "classic pizza base",  "calories": 800 },
    { "ident": "def", "description": "sliced salami",       "calories": 320 },
]);

let mut g = Graph::new();
g.add_node("order",     order_json);
g.add_node("catalogue", catalogue);

// Resolve each item's ident against the catalogue
let result = g.query(">/order/..items/#resolve('ident', >/catalogue, 'ident')")?;
```

---

## Enterprise example

The following dataset models a B2B SaaS company's operational data — customers, product catalogue, sales orders, line items, and employees — all in a single document. Every Jetro feature is demonstrated against it.

<details>
  <summary>Full JSON dataset (click to expand)</summary>

```json
{
  "company": {
    "name": "Arctiq Systems Inc.",
    "founded": 2012,
    "industry": "Enterprise SaaS",
    "headquarters": "San Francisco, CA",
    "regions": ["NA", "EMEA", "APAC"]
  },
  "customers": [
    {
      "id": "cust-001",
      "name": "Helios Corp",
      "tier": "enterprise",
      "region": "NA",
      "active": true,
      "contact": { "email": "procurement@helios.io", "phone": "+1-415-555-0101" },
      "contract": { "mrr": 12000, "seats": 200, "renewal": "2026-01-15" },
      "tags": ["strategic", "upsell"]
    },
    {
      "id": "cust-002",
      "name": "Nova Analytics",
      "tier": "mid-market",
      "region": "EMEA",
      "active": true,
      "contact": { "email": "ops@novaanalytics.eu", "phone": "+44-20-5555-0182" },
      "contract": { "mrr": 3400, "seats": 45, "renewal": "2025-11-30" },
      "tags": ["data-heavy", "expansion"]
    },
    {
      "id": "cust-003",
      "name": "Stratum Finance",
      "tier": "enterprise",
      "region": "NA",
      "active": true,
      "contact": { "email": "it@stratumfinance.com", "phone": "+1-212-555-0150" },
      "contract": { "mrr": 18500, "seats": 350, "renewal": "2026-06-01" },
      "tags": ["strategic", "regulated"]
    },
    {
      "id": "cust-004",
      "name": "Pinewave Retail",
      "tier": "smb",
      "region": "APAC",
      "active": false,
      "contact": { "email": "admin@pinewave.com.au", "phone": "+61-2-5550-0234" },
      "contract": { "mrr": 890, "seats": 12, "renewal": "2025-09-01" },
      "tags": ["at-risk"]
    },
    {
      "id": "cust-005",
      "name": "Meridian Health",
      "tier": "mid-market",
      "region": "NA",
      "active": true,
      "contact": { "email": "procurement@meridianhealth.org", "phone": "+1-312-555-0199" },
      "contract": { "mrr": 5200, "seats": 80, "renewal": "2026-03-15" },
      "tags": ["regulated", "expansion"]
    }
  ],
  "products": [
    {
      "id": "prod-core",
      "name": "Arctiq Platform Core",
      "category": "platform",
      "unit_price": 60.00,
      "billing": "per_seat_monthly",
      "min_seats": 10
    },
    {
      "id": "prod-analytics",
      "name": "Advanced Analytics Add-on",
      "category": "add-on",
      "unit_price": 18.00,
      "billing": "per_seat_monthly",
      "min_seats": 1
    },
    {
      "id": "prod-sso",
      "name": "SSO & Compliance Pack",
      "category": "security",
      "unit_price": 1200.00,
      "billing": "flat_monthly",
      "min_seats": null
    },
    {
      "id": "prod-support",
      "name": "Premier Support",
      "category": "support",
      "unit_price": 2500.00,
      "billing": "flat_monthly",
      "min_seats": null
    },
    {
      "id": "prod-onboarding",
      "name": "Dedicated Onboarding",
      "category": "professional_services",
      "unit_price": 8000.00,
      "billing": "one_time",
      "min_seats": null
    }
  ],
  "orders": [
    {
      "id": "ord-1001",
      "customer_id": "cust-001",
      "rep_id": "rep-a",
      "status": "closed_won",
      "created_at": "2025-01-10",
      "total": 15700.00,
      "line_items": [
        { "product_id": "prod-core",    "qty": 200, "unit_price": 60.00,   "subtotal": 12000.00 },
        { "product_id": "prod-sso",     "qty": 1,   "unit_price": 1200.00, "subtotal": 1200.00  },
        { "product_id": "prod-support", "qty": 1,   "unit_price": 2500.00, "subtotal": 2500.00  }
      ]
    },
    {
      "id": "ord-1002",
      "customer_id": "cust-003",
      "rep_id": "rep-b",
      "status": "closed_won",
      "created_at": "2025-02-14",
      "total": 27275.00,
      "line_items": [
        { "product_id": "prod-core",      "qty": 350, "unit_price": 60.00,   "subtotal": 21000.00 },
        { "product_id": "prod-analytics", "qty": 350, "unit_price": 14.50,   "subtotal": 5075.00  },
        { "product_id": "prod-sso",       "qty": 1,   "unit_price": 1200.00, "subtotal": 1200.00  }
      ]
    },
    {
      "id": "ord-1003",
      "customer_id": "cust-002",
      "rep_id": "rep-a",
      "status": "pending",
      "created_at": "2025-03-05",
      "total": 4080.00,
      "line_items": [
        { "product_id": "prod-core",        "qty": 45, "unit_price": 60.00,  "subtotal": 2700.00 },
        { "product_id": "prod-analytics",   "qty": 45, "unit_price": 15.00,  "subtotal": 675.00  },
        { "product_id": "prod-onboarding",  "qty": 1,  "unit_price": 705.00, "subtotal": 705.00  }
      ]
    },
    {
      "id": "ord-1004",
      "customer_id": "cust-005",
      "rep_id": "rep-b",
      "status": "pending",
      "created_at": "2025-03-20",
      "total": 6300.00,
      "line_items": [
        { "product_id": "prod-core",    "qty": 80, "unit_price": 60.00,   "subtotal": 4800.00 },
        { "product_id": "prod-sso",     "qty": 1,  "unit_price": 1200.00, "subtotal": 1200.00 },
        { "product_id": "prod-support", "qty": 1,  "unit_price": 300.00,  "subtotal": 300.00  }
      ]
    },
    {
      "id": "ord-1005",
      "customer_id": "cust-004",
      "rep_id": "rep-c",
      "status": "closed_lost",
      "created_at": "2025-02-28",
      "total": 720.00,
      "line_items": [
        { "product_id": "prod-core", "qty": 12, "unit_price": 60.00, "subtotal": 720.00 }
      ]
    }
  ],
  "employees": [
    {
      "id": "rep-a",
      "name": "Sandra Cole",
      "role": "account_executive",
      "region": "NA",
      "quota": 80000,
      "active": true
    },
    {
      "id": "rep-b",
      "name": "Marcus Reyes",
      "role": "account_executive",
      "region": "NA",
      "quota": 90000,
      "active": true
    },
    {
      "id": "rep-c",
      "name": "Yuki Tanaka",
      "role": "account_executive",
      "region": "APAC",
      "quota": 50000,
      "active": true
    }
  ]
}
```
</details>

---

### Path navigation

**Company name (simple child access):**
```
>/company/name
```
```json
"Arctiq Systems Inc."
```

---

**All customer contact emails (recursive descendant):**
```
>/customers/..email
```
```json
["procurement@helios.io", "ops@novaanalytics.eu", "it@stratumfinance.com",
 "admin@pinewave.com.au", "procurement@meridianhealth.org"]
```

---

**First matching top-level key (alternatives):**
```
>/('billing_contact' | 'company')/name
```
```json
"Arctiq Systems Inc."
```

---

**Recursive search for all pending orders:**
```
>/..('status'='pending')
```
```json
[
  { "id": "ord-1003", "customer_id": "cust-002", "status": "pending", ... },
  { "id": "ord-1004", "customer_id": "cust-005", "status": "pending", ... }
]
```

---

**First three orders (slice):**
```
>/orders/[:3]
```

---

### Aggregation & math

**Total pipeline value across all orders:**
```
>/orders/..subtotal/#sum
```
```json
49450.0
```

---

**Average monthly recurring revenue:**
```
>/customers/..mrr/#avg
```
```json
7998.0
```

---

**Largest single order value:**
```
>/orders/..total/#max
```
```json
27275.0
```

---

**Apply 10% enterprise discount to each product price:**
```
>/products/#map(>{ "name": @/name, "discounted": @/unit_price })
```
*(pair with `#mul(0.9)` per field in a full pipeline)*

---

### Filtering

**All closed-won orders:**
```
>/orders/#filter('status' == 'closed_won')
```

---

**Enterprise customers in NA region:**
```
>/customers/#filter('tier' == 'enterprise' and 'region' == 'NA')
```
```json
[
  { "id": "cust-001", "name": "Helios Corp",     "tier": "enterprise", "region": "NA", ... },
  { "id": "cust-003", "name": "Stratum Finance", "tier": "enterprise", "region": "NA", ... }
]
```

---

**First active enterprise customer (`#find`):**
```
>/customers/#find('tier' == 'enterprise')
```
```json
{ "id": "cust-001", "name": "Helios Corp", "tier": "enterprise", ... }
```

---

**All customers tagged as regulated (`#filter_by`):**
```
>/customers/#filter_by('tier' == 'mid-market')
```
```json
[
  { "id": "cust-002", "name": "Nova Analytics", ... },
  { "id": "cust-005", "name": "Meridian Health", ... }
]
```

---

### Grouping & counting

**Orders grouped by status:**
```
>/orders/#group_by('status')
```
```json
{
  "closed_lost": [ { "id": "ord-1005", ... } ],
  "closed_won":  [ { "id": "ord-1001", ... }, { "id": "ord-1002", ... } ],
  "pending":     [ { "id": "ord-1003", ... }, { "id": "ord-1004", ... } ]
}
```

---

**Count orders per status:**
```
>/orders/#count_by('status')
```
```json
{ "closed_lost": 1, "closed_won": 2, "pending": 2 }
```

---

**Customer tier distribution (`#tally` on plucked values):**
```
>/customers/#pluck('tier')/#tally
```
```json
{ "enterprise": 2, "mid-market": 2, "smb": 1 }
```

---

**Index customers by ID for O(1) lookup:**
```
>/customers/#index_by('id')
```
```json
{
  "cust-001": { "id": "cust-001", "name": "Helios Corp", ... },
  "cust-002": { "id": "cust-002", "name": "Nova Analytics", ... },
  ...
}
```

---

### Sorting & extracting

**Products sorted by unit price descending:**
```
>/products/#sort_by('unit_price', 'desc')
```

---

**All customer names (pluck):**
```
>/customers/#pluck('name')
```
```json
["Helios Corp", "Nova Analytics", "Stratum Finance", "Pinewave Retail", "Meridian Health"]
```

---

**Distinct billing models across all products:**
```
>/products/#pluck('billing')/#unique
```
```json
["flat_monthly", "one_time", "per_seat_monthly"]
```

---

**Orders chunked into batches of 2:**
```
>/orders/#chunk(2)
```
```json
[
  [ { "id": "ord-1001", ... }, { "id": "ord-1002", ... } ],
  [ { "id": "ord-1003", ... }, { "id": "ord-1004", ... } ],
  [ { "id": "ord-1005", ... } ]
]
```

---

### Construction literals

**Build an executive summary object from expressions:**
```
>{
  "company":        >/company/name,
  "total_customers": >/customers/#len,
  "active_mrr":     >/customers/#filter('active' == true)/..mrr/#sum,
  "open_pipeline":  >/orders/#filter('status' == 'pending')/..total/#sum
}
```
```json
{
  "company": "Arctiq Systems Inc.",
  "total_customers": 5,
  "active_mrr": 39590,
  "open_pipeline": 10380.0
}
```

---

**Build a flat array of employee names:**
```
>[ >/employees/[0]/name, >/employees/[1]/name, >/employees/[2]/name ]
```
```json
["Sandra Cole", "Marcus Reyes", "Yuki Tanaka"]
```

---

**Reshape each order using `@` current-item path:**
```
>/orders/#map(>{
  "order_ref":  @/id,
  "customer":   @/customer_id,
  "rep":        @/rep_id,
  "value":      @/total,
  "outcome":    @/status
})
```
```json
[
  { "order_ref": "ord-1001", "customer": "cust-001", "rep": "rep-a", "value": 15700.0, "outcome": "closed_won"  },
  { "order_ref": "ord-1002", "customer": "cust-003", "rep": "rep-b", "value": 27275.0, "outcome": "closed_won"  },
  { "order_ref": "ord-1003", "customer": "cust-002", "rep": "rep-a", "value": 4080.0,  "outcome": "pending"     },
  { "order_ref": "ord-1004", "customer": "cust-005", "rep": "rep-b", "value": 6300.0,  "outcome": "pending"     },
  { "order_ref": "ord-1005", "customer": "cust-004", "rep": "rep-c", "value": 720.0,   "outcome": "closed_lost" }
]
```

---

### Object manipulation

**Strip internal tags from a customer record:**
```
>/customers/[0]/#omit('tags')
```

---

**Keep only the contract sub-object:**
```
>/customers/[0]/#select('contract')
```
```json
{ "contract": { "mrr": 12000, "seats": 200, "renewal": "2026-01-15" } }
```

---

**Rename `mrr` to `monthly_revenue` in each contract:**
```
>/customers/..contract/#rename('mrr', 'monthly_revenue')
```

---

**Merge all employee records into one object:**
```
>/employees/#map(>{ [>/employees/#pluck('id')]: @/name })/#merge
```

---

**Remove null `min_seats` from every product:**
```
>/products/#map(>{ "id": @/id, "name": @/name, "price": @/unit_price, "billing": @/billing })
```

---

### Join & lookup

**Enrich every order with its full customer record (inner join on `id`):**
```
>/orders/#join(>/customers, 'customer_id', 'id')
```
```json
[
  { "id": "ord-1001", "customer_id": "cust-001", "name": "Helios Corp",     "tier": "enterprise", "total": 15700.0, ... },
  { "id": "ord-1002", "customer_id": "cust-003", "name": "Stratum Finance", "tier": "enterprise", "total": 27275.0, ... },
  ...
]
```

---

**Left-join orders to customers (null-patched on miss):**
```
>/orders/#lookup(>/customers, 'customer_id', 'id')
```

---

### Field resolution

**Resolve `customer_id` to a full customer object on every order (`#resolve`):**
```
>/orders/#resolve('customer_id', >/customers, 'id')
```
```json
[
  {
    "id": "ord-1001",
    "customer_id": { "id": "cust-001", "name": "Helios Corp", "tier": "enterprise", ... },
    "total": 15700.0,
    ...
  },
  ...
]
```

---

**Dereference a specific order's customer ID to a full object (`#deref`):**
```
>/orders/[0]/customer_id/#deref(>/customers, 'id')
```
```json
{ "id": "cust-001", "name": "Helios Corp", "tier": "enterprise", "region": "NA", ... }
```

---

**Extract all line-item product IDs from a single order:**
```
>/orders/[0]/line_items/#pluck('product_id')
```
```json
["prod-core", "prod-sso", "prod-support"]
```

---

### Flatten & zip

**Collect every line item across all orders into a single flat array:**
```
>/orders/#map(>[ @/line_items ])/#flatten
```

---

**Zip rep names with their quotas:**
```
>/#pick(>/employees/#pluck('name') as 'rep', >/employees/#pluck('quota') as 'quota')/#zip
```
```json
[
  { "rep": "Sandra Cole",  "quota": 80000 },
  { "rep": "Marcus Reyes", "quota": 90000 },
  { "rep": "Yuki Tanaka",  "quota": 50000 }
]
```

---

### `#pick` with sub-expressions

**Build a full revenue dashboard in a single expression:**
```
>/#pick(
  >/orders/#filter('status' == 'closed_won')/..total/#sum   as 'closed_revenue',
  >/orders/#filter('status' == 'pending')/..total/#sum      as 'pipeline_value',
  >/customers/#filter('tier' == 'enterprise')/..mrr/#sum    as 'enterprise_mrr',
  >/customers/#len                                          as 'total_accounts',
  >/orders/#filter('status' == 'closed_won')/#len           as 'deals_closed'
)
```
```json
{
  "closed_revenue":  42975.0,
  "pipeline_value":  10380.0,
  "enterprise_mrr":  30500,
  "total_accounts":  5,
  "deals_closed":    2
}
```

---

### Graph — cross-document queries

Load orders and customers as separate Graph nodes and query across them without embedding one document inside the other.

```rust
use jetro::graph::Graph;

let mut g = Graph::new();
g.add_node("orders",    orders_json);    // the "orders" array above
g.add_node("customers", customers_json); // the "customers" array above
g.add_node("products",  products_json);  // the "products" array above

// Revenue per customer tier — join happens across node boundaries
let result = g.query(
    ">/orders/#join(>/customers, 'customer_id', 'id')/#group_by('tier')"
)?;

// Build a report schema — each value is a Jetro expression evaluated
// against the merged virtual root
let report = g.message(r#"{
  "total_closed_won":    ">/orders/#filter('status' == 'closed_won')/..total/#sum",
  "enterprise_accounts": ">/customers/#filter('tier' == 'enterprise')/#len",
  "avg_deal_size":       ">/orders/#filter('status' == 'closed_won')/..total/#avg",
  "top_product":         ">/orders/..line_items/#flatten/#sort_by('subtotal', 'desc')/[0]/product_id"
}"#)?;
```

```json
{
  "total_closed_won":     42975.0,
  "enterprise_accounts":  2,
  "avg_deal_size":        21487.5,
  "top_product":          "prod-core"
}
```

---

## Architecture

Jetro consists of three layers:

- **Parser** (`src/parser.rs`) — PEG grammar (`grammar.pest`) parsed with [pest](https://pest.rs), produces a `Vec<Filter>`.
- **Context** (`src/context.rs`) — stack-based depth-first evaluator; each `StackItem` carries the current value and the remaining filter tail.
- **Functions** (`src/func.rs`) — pluggable `Callable` trait; `FuncRegistry` maps names to implementations. The default registry includes all built-in functions.
- **Graph** (`src/graph.rs`) — multi-document virtual root; all named nodes are merged so cross-document expressions work without any special syntax.
