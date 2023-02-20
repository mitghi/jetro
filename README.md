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

produces
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
---

```json
>/pick('foo', >/..person/formats('Herrn {} {}', 'firstname', 'lastname') as 'fullname'/fullname as 'fullname')
```

produces
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

---

```json
>/pick('friend', >/..person/formats('Herrn {} {}', 'firstname', 'lastname') as 'fullname'/fullname as 'fullname', >/foo/..contract)
```

produces
```json
{
  "friend": "Thunder Pur",
  "fullname": "Herrn Mio Snuggle",
  "kind": "Furry Purr"
}
```

---

```
>/..foo/[:4]
```

produces
```json
[
  1,
  2,
  3,
  4
]
```

---

```json
>/..meows/[4:]
```

produces
```json
[
  50,
  60
]
```
