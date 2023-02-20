# jetro

Jetro is a tool for transforming, querying and comparing data in JSON format.

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

# example

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
      "lastname": "Snuggle"
    }
  },
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
>/#pick('foo', >/..person/#formats('Herrn {} {}', 'firstname', 'lastname') as 'fullname'/fullname as 'fullname')
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
>/#pick('friend', >/..person/#formats('Herrn {} {}', 'firstname', 'lastname') as 'fullname'/fullname as 'fullname', >/foo/..contract)
```

<details>
  <summary>See output</summary>
  
  ### result

```json
{
  "friend": "Thunder Pur",
  "fullname": "Herrn Mio Snuggle",
  "kind": "Furry Purr"
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
>/..foo/..contract/#pick('kind' as 'contract', </..person/#formats('Welcome {}', 'firstname') as 'welcome_message'/#pick('welcome_message'))
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
