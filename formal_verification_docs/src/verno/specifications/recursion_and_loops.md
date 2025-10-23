# Recursion and loops

Constrained Noir keeps many loop proofs lightweight thanks to fixed bounds and deterministic control flow. As a result, simple patterns often verify without extra annotations. When additional reasoning is required, Verno now supports explicit [loop invariants](https://viperproject.github.io/prusti-dev/user-guide/tour/loop_invariants.html) and [decreases clauses](https://verus-lang.github.io/verus/guide/reference-decreases.html?#the-decreases-measure) via helpers in `fv_std`.

## Why do simple loops verify easily?

   1. **No recursion in constrained Noir** – Since recursion is disallowed in the constrained subset, termination proofs rarely come up.
   2. **Bounded loops** – Loop bounds are explicit, so even without decreases clauses the verifier can reason about termination.

When you step outside these patterns—e.g. working with `unconstrained fn`, ghost helpers, or more precise arithmetic—you can attach invariants and decreases clauses just like in other verification systems. See the [Unconstrained Noir Support](../unconstrained_support.md) guide for a full example.

## Example: Verifying a loop

The following Noir function increments `sum` in a loop and already verifies without extra annotations:
```rust,ignore
#['requires((0 <= x) & (x <= y) 
    & (y < 1000000))] // Prevent overflow when adding numbers
fn main(x: u32, y: u32) {
    let mut sum = y;
    for i in 0..100 {
        sum += x;
    }
    assert(sum >= y);
}
```
Since `sum` is always increasing and `x` is non-negative, the assertion `sum >= y` holds for all valid inputs. If you strengthen the postconditions or need to expose intermediate facts, add `invariant(...)` clauses (and `decreases(...)` when necessary) to the loop body.

### Recursion

In Verno you *can* prove statements for recursive functions but this is not beneficial because programs with recursion **can not** be built by the standard Noir compiler.
