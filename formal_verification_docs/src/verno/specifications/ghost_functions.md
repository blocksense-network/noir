# Ghost Functions
In many formal verification systems, there is a strict separation between code for mathematical proofs and for execution.
Some systems, like **Verus**, introduce explicit "ghost functions"—special functions used purely for proofs and omitted from compiled code.
Others, like **Prusti**, allow calling regular functions inside specifications without requiring them to be ghost functions.

**Verno supports ghost functions**.

A function can be marked as `#['ghost]` to indicate that it is only for use in specifications. Ghost functions are not compiled to ACIR and do not affect the final program, so they can be used to write verification logic that would be too expensive or impossible to run in a real execution.

This is useful for defining helper functions for your specifications that are not part of the program's core logic.

### Example from Test Suite

The project's test suite includes several examples of ghost functions. While most are minimal, the following example, adapted from the `is_power_of_2` test, is the most illustrative.

It uses a ghost function to define a reusable check, which keeps the specifications for the main logic clean and readable.

```rust,ignore
#['ghost]
fn is_power_of_2(x: i32) -> bool {
  if x <= 0 {
      false
    } else {
      (x & (x - 1)) == 0
    }
}

#['requires(is_power_of_2(x))]
#['requires(is_power_of_2(y))]
#['ensures(is_power_of_2(result))]
fn multiply_powers_of_2(x: i32, y: i32) -> pub i32 {
    x * y
}
```
In this example, `is_power_of_2` is a ghost function used in the `requires` and `ensures` annotations to verify the `multiply_powers_of_2` function. The `is_power_of_2` function itself is not part of the compiled program. This helps to keep the specification of `multiply_powers_of_2` clean and readable.
The traditional approach of using ghost functions, proofs, and lemmas has its benefits, especially for reasoning about complex systems like distributed systems.
However, it also has drawbacks, such as reduced code reusability and increased mathematical complexity, which can make the verification process less user-friendly.

