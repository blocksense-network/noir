# Recursion and loops

Noir FV streamlines formal verification by eliminating the need for [loop invariants](https://viperproject.github.io/prusti-dev/user-guide/tour/loop_invariants.html) and [termination proofs](https://verus-lang.github.io/verus/guide/recursion.html).  
Unlike traditional frameworks that rely on techniques like induction, decrease clauses, and invariants to prove correctness for recursive functions and loops, constrained Noir avoids these complexities altogether.

## Why?

   1. **No Recursion** – Noir does not support recursion, eliminating the need for termination proofs.
   2. **Bounded Loops** – All loops in constrained Noir have fixed bounds, meaning they always terminate.

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
