# Limitations

Verno still omits a handful of Noir features. The following constructs remain unsupported today:

* **Lambda functions**
* **Standard library functions** that rely on runtime helpers beyond the formal-verification shim
* **Vectors** (`Vec<T>`)
* **Optional types** (`Option<T>`)

These limitations may change as the toolchain evolves, so check this page when upgrading to a new release. The examples below illustrate the current verifier behaviour when the unsupported features are used.

## Lambda Functions
```rust,ignore
#['ensures(result == x / 2)]
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
1 │ #['ensures(result == x / 2)]
  │           ---------------- failed this postcondition
  │

Error: Verification failed!
```

## Standard Library Functions
```rust,ignore
#['requires(x.lt(y))]
#['ensures(result == x)]
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

## Recently Lifted Limitations

Two previously missing capabilities are now available:

- **Mutable references** are supported as function parameters. When writing specifications against mutable borrows, use `fv_std::old()` to refer to the incoming value (see the ghost functions guide for examples).
- **Unconstrained Noir functions** participate in verification, provided their bodies introduce the right `#[requires]`/`#[ensures]` contracts and loop annotations (`invariant`, `decreases`, etc.). The dedicated *Unconstrained Noir Support* page outlines the required proof obligations.

Although some of the unsupported features above might appear straightforward, we continue to prioritize end-to-end verification of core Noir programs before expanding the surface area further.
