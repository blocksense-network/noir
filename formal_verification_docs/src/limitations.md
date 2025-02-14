# Limitations

As a prototype, Noir FV has certain limitations and does not yet support some features of Noir. The following features are currently unsupported:
* **Mutable references** as function parameters
* **Lambda functions**
* **Standard library functions**
* **Unconstrained functions**
* **Vectors** (`Vec<T>`)
* **Optional types** (`Option<T>`)

These limitations may change as Noir FV evolves.

These examples show the errors produced when using unsupported features in Noir FV.

## Non-Local Mutable References
```rust,ignore
fn main() {
    let mut x = 5;
    nullify(&mut x);
    assert(x == 0);
}

fn nullify(x: &mut u32) {
  *x = 0;
}
```
Output after running `nargo fv`:
```
Error: Verification crashed!
```

## Lambda Functions
```rust,ignore
#[ensures(result == x / 2)]
fn main(x: u32) -> pub u32 {
  let divide_by_2 = |val| val / 2; // Lambda function
  divide_by_2(x)
}
```
Output:
```
error: postcondition not satisfied
  ┌─ src/main.nr:1:11
  │
1 │ #[ensures(result == x / 2)]
  │           ---------------- failed this postcondition
  │

Error: Verification failed!
```

## Standard Library Functions
```rust,ignore
#[requires(x.lt(y))]
#[ensures(result == x)]
fn main(x: Field, y: pub Field) -> pub Field {
    if x.lt(y) {
      x
    } else {
      y
    }
}
```
Output:
```
The application panicked (crashed).
Message:  called `Option::unwrap()` on a `None` value
```

## Unconstrained Functions
```rust,ignore
#[requires(num < 100000)]
#[ensures(result == num + 1)]
fn main(num: u32) -> pub u32 {
    let out = unsafe { 
        increment(num) 
    };
    out
}
#[requires(num < 100000)]
#[enusres(result == num + 1)]
unconstrained fn increment(num: u32) -> u32 {
    if num > 100000 {
      0
    } else {
      num + 1
    }
}
```
Output:
```
The application panicked (crashed).
Message:  internal error: entered unreachable code
```
## Vectors
```rust,ignore
fn main(x: Field, y: pub Field) {
  let mut vector: Vec<Field> = Vec::new();
  for i in 0..5 {
    vector.push(i as Field);
  }
  assert(vector.len() == 5);
}
```
Output:
```
The application panicked (crashed).
Message:  internal error: entered unreachable code
```

## Optional Types
```rust,ignore
fn main() {
    let none: Option<u32> = Option::none();
    assert(none.is_none());
    let some = Option::some(3);
    assert(some.unwrap() == 3);
}
```
Output:
```
error: assertion failed
  ┌─ src/main.nr:5:12
  │
5 │     assert(some.unwrap() == 3);
  │            ------------------- assertion failed
  │

Error: Verification failed!
```