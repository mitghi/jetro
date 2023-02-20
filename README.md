# jetro

Jetro is a tool for transforming, querying and comparing data in JSON format.

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

```json
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

```json
>/pick('foo', >/..person/formats('Herrn {} {}', 'firstname', 'lastname') as 'fullname'/fullname as 'fullname')
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

```json
>/pick('friend', >/..person/formats('Herrn {} {}', 'firstname', 'lastname') as 'fullname'/fullname as 'fullname', >/foo/..contract)
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

```json
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
