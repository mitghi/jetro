# jetro

[<img src="https://img.shields.io/badge/docs-jetro-blue"></img>](https://docs.rs/jetro)
[<img src="https://img.shields.io/badge/try-online%20repl-brightgreen"></img>](https://jetro.io)
![GitHub](https://img.shields.io/github/license/mitghi/jetro)

Jetro is a library which provides a custom DSL for transforming, querying and comparing data in JSON format. It is easy to use and extend.

Jetro has minimal dependency, the traversal and eval algorithm is implemented on top of [serde_json](https://serde.rs).

Jetro can be used inside Web Browser by compiling down to WASM. Visit [Jetro Web](https://jetro.io)
to try it online, or [clone it](https://github.com/mitghi/jetroweb) and give it a shot.

Jetro can be used in command line using [Jetrocli](https://github.com/mitghi/jetrocli).

Jetro combines access path with functions which operate on values matched within the pipeline. Access path uses `/` as separator similar to structure of URI, the start of access path should denote whether the access starts from root by using `>`, it is possible to traverse from root in nested paths by using `<`. 

Jetro expressions support line breaks and whitespace, the statements can be broken up into smaller parts.

By convention, functions are denoted using `#` operator. Functions can be composed.

| Function | Action |
| -------- | ------ |
| #pick('string' \| expression, ...) [ as \| as* 'binding_value' ] | Select a key from an object, bind it to a name, select multiple sub queries to create new object |
| #head | Head of the list|
| #tail | Tail of the list |
| #keys | Keys associated with an object |
| #values | Values associated with an object |
| #reverse | Reverse the list |
| #min | Min value of numbers |
| #max | Max value of numbers |
| #all | Whether all boolean values are true |
| #sum | Sum of numbers |
| #formats('format with placeholder {} {}', 'key_a', 'key_b') [ -> \| ->* 'binding_value' ] | Insert formatted key:value into object or return it as single key:value  |
| #filter('target_key' (>, <, >=, <=, ==, ~=, !=) (string, boolean, number)) | Perform Filter on list |
| #map(x: x.y.z \| x.y.z.some_method())| Map each item in a list with the given lambda |


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
#### structure
Jetro consists of a parser, context wrapper which manages traversal and evaluation of each step of user input and a runtime for dynamic functions. The future version will support user-defined functions.

# example

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
      {
        "ident": "abc",
        "is_gratis": false,
        "name": "pizza",
        "price": 4.8,
        "total": 1,
        "type": "base_composable"
      },
      {
        "ident": "def",
        "is_gratis": false,
        "name": "salami",
        "price": 2.8,
        "total": 10,
        "type": "ingredient"
      },
      {
        "ident": "ghi",
        "is_gratis": false,
        "name": "cheese",
        "price": 2,
        "total": 1,
        "type": "ingredient"
      },
      {
        "ident": "uip",
        "is_gratis": true,
        "name": "chilli",
        "price": 0,
        "total": 1,
        "type": "ingredient"
      },
      {
        "ident": "ewq",
        "is_gratis": true,
        "name": "bread sticks",
        "price": 0,
        "total": 8,
        "type": "box"
      }
    ]
  }
}
```

### Queries

Get value associated with `line_items`.

```
>/line_items
```
<details>
  <summary>See output</summary>

  ### result
  ```json
{
  "items": [
    {
      "ident": "abc",
      "is_gratis": false,
      "name": "pizza",
      "price": 4.8,
      "total": 1,
      "type": "base_composable"
    },
    {
      "ident": "def",
      "is_gratis": false,
      "name": "salami",
      "price": 2.8,
      "total": 10,
      "type": "ingredient"
    },
    {
      "ident": "ghi",
      "is_gratis": false,
      "name": "cheese",
      "price": 2,
      "total": 1,
      "type": "ingredient"
    },
    {
      "ident": "uip",
      "is_gratis": true,
      "name": "chilli",
      "price": 0,
      "total": 1,
      "type": "ingridient"
    },
    {
      "ident": "ewq",
      "is_gratis": true,
      "name": "bread sticks",
      "price": 0,
      "total": 8,
      "type": "box"
    }
  ]
}
  ```
</details>

---

Get value associated with first matching key which has a value and return its `id` field.

```
>/('non-existing-member' | 'customer')/id
```

<details>
  <summary>See output</summary>

  ### result

```json
"xyz"
```
</details>

---
```
>/..items/#tail
```

<details>
  <summary>See output</summary>

  ### result

```json
[
  {
    "ident": "def",
    "is_gratis": false,
    "name": "salami",
    "price": 2.8,
    "total": 10,
    "type": "ingredient"
  },
  {
    "ident": "ghi",
    "is_gratis": false,
    "name": "cheese",
    "price": 2,
    "total": 1,
    "type": "ingredient"
  },
  {
    "ident": "uip",
    "is_gratis": true,
    "name": "chilli",
    "price": 0,
    "total": 1,
    "type": "ingridient"
  },
  {
    "ident": "ewq",
    "is_gratis": true,
    "name": "bread sticks",
    "price": 0,
    "total": 8,
    "type": "box"
  }
]
```
</details>

---

```
>/..items/#filter('is_gratis' == true and 'name' ~= 'ChILLi')
```

<details>
  <summary>See output</summary>

  ### result

```json
[
  {
    "ident": "uip",
    "is_gratis": true,
    "name": "chilli",
    "price": 0,
    "total": 1,
    "type": "ingridient"
  }
]
```
</details>

---

```
>/..items/#filter('is_gratis' == true and 'name' ~= 'ChILLi')/#map(x: x.type)
```

<details>
  <summary>See output</summary>

  ### result

```json
[
  "ingridient"
]
```
</details>

---

Create a new object with scheme `{'total': ..., 'fullname': ...}` as follow:
- recursively search for `line_items`, dive into any matched object, filter matches with `is_gratis == false` statement, recursively look for their prices and return the sum of prices
- recursively search for object `user`, select its `profile` and create a new object with schema `{'fullname': ...}` formated by concatenating values of keys ('firstname', 'lastname')

```
>/#pick(
  >/..line_items
   /*
   /#filter('is_gratis' == false)/..price/#sum as 'total',

  >/..user
   /profile
   /#formats('{} {}', 'firstname', 'lastname') ->* 'fullname'
)
```

<details>
  <summary>See output</summary>

  ### result

```json
{
  "fullname": "John Appleseed",
  "total": 9.6
}
```
</details>

---

Select up to 4 items from index zero of array `items`

```
>/..items/[:4]
```

<details>
  <summary>See output</summary>

  ### result

```json
[
  {
    "ident": "abc",
    "is_gratis": false,
    "name": "pizza",
    "price": 4.8,
    "total": 1,
    "type": "base_composable"
  },
  {
    "ident": "def",
    "is_gratis": false,
    "name": "salami",
    "price": 2.8,
    "total": 10,
    "type": "ingredient"
  },
  {
    "ident": "ghi",
    "is_gratis": false,
    "name": "cheese",
    "price": 2,
    "total": 1,
    "type": "ingredient"
  },
  {
    "ident": "uip",
    "is_gratis": true,
    "name": "chilli",
    "price": 0,
    "total": 1,
    "type": "ingridient"
  }
]
```
</details>

---

Select from 4th index and consume until end of array `items`

```
>/..items/[4:]
```

<details>
  <summary>See output</summary>

  ### result

```json
[
  {
    "ident": "ewq",
    "is_gratis": true,
    "name": "bread sticks",
    "price": 0,
    "total": 8,
    "type": "box"
  }
]
```
</details>

---

Create a new object with schema `{'total_gratis': ...}` as follow:
- Recursively look for any object containing `items`, and then recursively search within the matched object for `is_gratis` and length of matched values

```
>/#pick(>/..items/..is_gratis/#len as 'total_gratis')
```

<details>
  <summary>See output</summary>

  ### result

```json
{
  "total_gratis": 2
}
```
</details>

---

Recursively search for object `items`, select its first item and return its keys

```
>/..items/[0]/#keys
```

<details>
  <summary>See output</summary>

  ### result

```json
[
  "ident",
  "is_gratis",
  "name",
  "price",
  "total",
  "type"
]
```
</details>

---

Recursively search for object `items`, select its first item and return its values

```
>/..items/[0]/#values
```

<details>
  <summary>See output</summary>

  ### result

```json
[
  "abc",
  false,
  "pizza",
  4.8,
  1,
  "base_composable"
]
```
</details>

