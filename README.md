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
      "lastname": "Snuggle",
	  "hasFurr": true
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
