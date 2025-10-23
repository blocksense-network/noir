# Specification syntax

Verno specifications are a superset of Noir's Boolean expressions. The key extensions are summarized below:

| Syntax | Meaning |
| ------ | ------- |
| [`result`](#result-variable) | Return value of the function |
| [`... ==> ...`](specifications/quantifiers.md#logical-implication) | Right implication |
| [`forall(...)`](specifications/quantifiers.md#forall) | Universal quantifier |
| [`exists(...)`](specifications/quantifiers.md#exists) | Existential quantifier |
| [`old(expr)`](#old-helper) | Value of `expr` at function/loop entry |
| [`#['ghost]`](specifications/ghost_functions.md) | Declares a ghost-only helper |

## `result` Variable

In Verno, the special variable `result` refers to the return value of a function. It can only be used in **postconditions** (`ensures` annotations) and must not be used as a regular function parameter name.

Here is an example for returning an integer:
```rust,ignore
#['ensures(result == 8)]
fn eight() -> i32 {
    8
}
```
And an example for returning a tuple and accessing individual fields:
```rust,ignore
#['ensures((result.0 / -2 == result.1.0 as i32) & (result.1.1 == 0))]
fn main() -> pub (i32, (u32, u32)) {
    (-8, (4, 0))
}
```

## Noir Boolean Expressions

Noir does not support the standard logical operators `||` (OR) and `&&` (AND). Instead, it uses the bitwise operators `|` and `&`, which function the same way for Boolean values but do not short-circuit.

One implication of this is the need for additional parentheses to ensure the correct order of operations, as bitwise operators follow different precedence rules.

## Type Checking in Annotations

Verno enforces type checking within annotation expressions. While this ensures consistency, it may sometimes produce errors that seem unintuitive, especially to users familiar with other formal verification systems.

In some cases, you may need to add explicit type casts to satisfy the type checker. When doing so, always prefer **upcasting** (from smaller integer type to larger one) rather than **downcasting** (from larger integer type to smaller one), as downcasting can result in loss of information, leading to incorrect verification results. While downcasting may allow the type checker to accept an expression, it can introduce unintended behavior that prevents the verifier from proving correctness.

### Example: Why Upcasting is Necessary
Consider the following function::
```rust,ignore
#['requires(x >= y)]
#['ensures(result == y + 1)]
fn main(x: u16, y: u32) -> pub u32 {
    y + 1
}
```
Attempting to verify this code results in a type error:
```
error: Integers must have the same bit width LHS is 16, RHS is 32
  ┌─ src/main.nr:1:12
  │
1 │ #['requires(x >= y)]
  │            -----
  │

Error: Aborting due to 1 previous error
```
To fix the error, we might attempt a downcast, converting `y` to `u16`:
```rust,ignore
#['requires(x >= y as u16)]
#['ensures(result == y + 1)]
fn main(x: u16, y: u32) -> pub u32 {
    y + 1
}
```
However, this leads to a formal verification failure:
```
error: possible arithmetic underflow/overflow
  ┌─ src/main.nr:4:3
  │  
4 │ ╭   y + 1
5 │ │ }
  │ ╰'
  │  

Error: Verification failed!
```
This failure is expected. If `y` were `max(u32) = 4,294,967,295`, downcasting it to `u16` would result in `y = 65,535`, which is the maximum possible `u16` value. This allows for a scenario where `x = 65,535` and `y = 4,294,967,295`, making the precondition (`x >= y as u16`) true. However, adding `1` to `y` would cause an overflow.

Instead of downcasting `y`, the correct approach is to **upcast** `x` to match `y`'s bit width::
```rust,ignore
#['requires(x as u32 >= y)]
#['ensures(result == y + 1)]
fn main(x: u16, y: u32) -> pub u32 {
    y + 1
}
```
After running `verno formal-verify`:
```
Verification successful!
```
## `old` Helper

The `old(expr)` helper refers to the value of an expression at the start of a function (inside `ensures`) or at the start of each loop iteration (inside an `invariant`). This is particularly useful when specifying the behaviour of mutable references.

```rust,ignore
use fv_std::old;

#['requires(*old(counter) < u32::MAX)]
#['ensures(*counter == *old(counter) + 1)]
fn bump(counter: &mut u32) {
    *counter += 1;
}
```

The `old` value is read-only: you can use it in specifications, but not assign to it.

## Ghost Functions

Functions annotated with `#['ghost]` are excluded from the compiled circuit and exist solely for specifications. They keep proof-oriented helpers separate from executable code.

```rust,ignore
#['ghost]
fn is_power_of_2(x: i32) -> bool {
    x > 0 & ((x & (x - 1)) == 0)
}

#['requires(is_power_of_2(x) & is_power_of_2(y))]
#['ensures(is_power_of_2(result))]
fn multiply_powers_of_2(x: i32, y: i32) -> pub i32 {
    x * y
}
```

Ghost helpers can call other ghost code and appear freely inside `requires`/`ensures` clauses.
