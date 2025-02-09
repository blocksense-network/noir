# Recursion and loops

Unlike many formal verification systems, Noir FV eliminates the need for [loop invariants](https://viperproject.github.io/prusti-dev/user-guide/tour/loop_invariants.html) and [termination proofs](https://verus-lang.github.io/verus/guide/recursion.html). In most verification frameworks, proving correctness for recursive functions and loops requires techniques like induction, decrease clauses, and loop invariants to ensure termination and correctness. However, in constrained Noir, these complexities are unnecessary.
## Why?

   1. **No Recursion** – Noir FV does not support recursion, eliminating the need for termination proofs.
   2. **Bounded Loops** – All loops in constrained Noir have fixed bounds, meaning they always terminate.

## Key Benefits

* **Induction is not mandatory** – In many cases, Noir FV avoids the need for induction by enforcing bounded loops and eliminating recursion.
* **No termination proofs** – Unlike many other verification systems, we don’t need explicit proofs that functions eventually return.
* **Simpler verification** – Noir FV statically verifies correctness without requiring extra annotations like invariants.

## Example: Verifying a Loop

The following Noir function increments `sum` in a loop and successfully verifies **without needing invariants**:
```rust,ignore
#[requires((0 <= x) & (x <= y) 
    & (y < 1000000))] // Prevent overflow when adding numbers
fn main(x: u32, y: u32) {
    let mut sum = y;
    for i in 0..100 {
        sum += x;
    }
    assert(sum >= y);
}
```
Since `sum` is always increasing and `x` is non-negative, the assertion `sum >= y` holds for all valid inputs. Noir FV can verify this automatically without requiring additional annotations.

## Summary

By **eliminating recursion and enforcing bounded loops**, Noir FV simplifies the verification process while ensuring rigorous correctness. This approach avoids the need for complex proof techniques, making formal verification **more accessible** without sacrificing **mathematical soundness**.