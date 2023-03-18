# jetro

[<img src="https://img.shields.io/badge/docs-jetro-blue"></img>](https://docs.rs/jetro)
[<img src="https://img.shields.io/badge/try-online%20repl-brightgreen"></img>](https://jetro.io)
![GitHub](https://img.shields.io/github/license/mitghi/jetro)

Jetro is a tool with custom DSL for transforming, querying and comparing data in JSON format.

Jetro has minimal dependency, the traversal and eval algorithm is implemented on top of [serde_json](https://serde.rs).

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

Jetro can be used in Web Browser by compiling down to WASM. Visit [Jetro Web](https://jetro.io)
to try it online.

# example

Jetro combines access path with functions which operate on those values matched within the pipeline.

By convention, functions are denoted using # operator. Functions can be composed.

| Function | Action |
| -------- | ------ |
| #pick('string' \| expression, ...) [ as \| as* 'binding_value' ] | Select a key from an object, bind it to a name, select multiple sub queries to create new object |
| #head | Head of the list|
| #tail | Tail of the list |
| #reverse | Reverse the list |
| #all | Whether all boolean values are true |
| #sum | Sum of numbers |
| #formats('format with placeholder {} {}', 'key_a', 'key_b') [ -> \| ->* 'binding_value' ] | Insert formatted key:value into object or return it as single key:value  |
| #filter('target_key' (>, <, >=, <=, ==, ~=, !=) (string, boolean, number)) | Perform Filter on list |

```json
{
  "bar": {
    "meows": [
      10,
      20,
      30,
      40,
      50,
      60
    ],
    "person": {
      "firstname": "Mio",
      "lastname": "Snuggle",
      "hasFurr": true
    },
    "rolls": [
      {
        "roll": "on side"
      },
      {
        "roll": "in space"
      },
      {
        "roll": "in multiverse"
      },
      {
        "roll": "everywhere"
      }
    ]
  },
  "foo": [
    1,
    2,
    3,
    4,
    5,
    {
      "contract": {
        "kind": "Furry Purr",
        "hasFurr": false
      }
    }
  ],
  "friend": "Thunder Pur"
}
```

### Queries

```
>/bar
```
<details>
  <summary>See output</summary>

  ### result
  ```json
  "bar": {
    "meows": [
      10,
      20,
      30,
      40,
      50,
      60
    ],
    "person": {
      "firstname": "Mio",
      "lastname": "Snuggle"
    }
  }
  ```
</details>

---
```
>/bar/('person' | 'whatever')
```

<details>
  <summary>See output</summary>

  ### result

```json
{
  "firstname": "Mio",
  "hasFurr": true,
  "lastname": "Snuggle"
}
```
</details>

---

```
>/('foo' | 'bar' | 'non-exinsting-key')
```

<details>
  <summary>See output</summary>

  ### result

```json
[
  1,
  2,
  3,
  4,
  5,
  {
    "contract": {
      "kind": "Furry Purr"
    }
  }
]
```
</details>

---

```
>/('foo' | 'bar' | 'non-exinsting-key')/[:5]/#sum
```

<details>
  <summary>See output</summary>

  ### result

```json
15
```
</details>

---

```
>/..rolls/#filter('roll' == 'everywhere')
```

<details>
  <summary>See output</summary>

  ### result

```json
[
  {
    "roll": "everywhere"
  }
]
```
</details>

---

```
>/..rolls/#filter('priority'  < 11 and 'roll' ~= 'ON Side')
```

<details>
  <summary>See output</summary>

  ### result

```json
[
  {
    "priority": 1,
    "roll": "on side"
  }
]
```
</details>

---

```
>/..rolls/#head/#formats('Rolling {}', 'roll') -> 'msg'
```

<details>
  <summary>See output</summary>

  ### result

```json
{
  "msg": "Rolling on side",
  "priority": 1,
  "roll": "on side"
}
```
</details>

---

```
>/..foo/[:4]
```

<details>
  <summary>See output</summary>

  ### result

```json
[
  1,
  2,
  3,
  4
]
```
</details>

---

```
>/..meows/[4:]
```

<details>
  <summary>See output</summary>

  ### result

```json
[
  50,
  60
]
```
</details>

---

```
>/#pick(>/..hasFurr/#all as 'totallyFurry')
```

<details>
  <summary>See output</summary>

  ### result

```json
{
  "totallyFurry": true
}
```
</details>

---

---

```
>/#pick('foo', >/..person/#formats('Herrn {} {}', 'firstname', 'lastname') ->* 'fullname')
```

<details>
  <summary>See output</summary>

  ### result

```json
{
  "foo": [
    1,
    2,
    3,
    4,
    5,
    {
      "contract": {
        "kind": "Furry Purr"
      }
    }
  ],
  "fullname": "Herrn Mio Snuggle"
}
```
</details>

---

```
>/..foo/..contract/#pick('kind' as 'contract', </..person/#formats('Welcome {}', 'firstname') ->* 'welcome_message')
```

<details>
  <summary>See output</summary>

  ### result

```json
{
  "contract": "Furry Purr",
  "welcome_message": "Welcome Mio"
}

```
</details>
