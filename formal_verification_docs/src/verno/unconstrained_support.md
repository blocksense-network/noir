# Unconstrained Noir Support

Verno now handles Noir programs that mix `unconstrained fn` bodies with specification annotations. This unlocks verification for idiomatic Noir code that performs side-effecting computation while remaining compatible with Noir's `unconstrained` execution model.

The key ingredients for a successful proof are:

1. Add the usual `#[requires(...)]` / `#[ensures(...)]` contracts to your functions, even if the function is marked `unconstrained`.
2. Annotate loops with `invariant(...)` (to preserve safety properties) and `decreases(...)` (to prove termination) from `fv_std`.
3. When mutating state through references, use `fv_std::old()` so specifications can refer to the pre-state.

The snippet below demonstrates the pattern on a simple accumulator:

```rust
use fv_std::{decreases, invariant, old};

#['ensures(result == slice_sum(arr, 0))]
unconstrained fn sum_array(arr: [u32; 8]) -> u32 {
    let mut acc = 0;
    let mut idx = 0;

    while idx < 8 {
        invariant(idx <= 8);
        invariant(acc == slice_sum(arr, idx));
        decreases(8 - idx);

        acc += arr[idx];
        idx += 1;
    }

    acc
}

#['requires(*old(out) == 0)]
#['ensures(*out == slice_sum(arr, 0))]
fn write_sum(out: &mut u32, arr: [u32; 8]) {
    let total = sum_array(arr);
    *out = total;
}

#['ghost]
fn slice_sum(arr: [u32; 8], start: u32) -> u32 {
    let mut sum = 0;
    let mut i = start;
    while i < 8 {
        invariant(start <= i & i <= 8);
        invariant(sum == arr[start..i].sum());
        decreases(8 - i);

        sum += arr[i];
        i += 1;
    }
    sum
}
```

This example showcases each of the supported ingredients:

- `sum_array` is `unconstrained`, yet its behaviour is described with a postcondition and loop obligations.
- Each `while` loop supplies both `invariant` and `decreases` clauses from `fv_std`.
- The `write_sum` wrapper uses `fv_std::old()` to relate the mutable reference `out` to its pre-state, an ability now supported across the toolchain.

### When to fall back to assumptions

If a proof step requires a fact that Noir's arithmetic reasoning cannot derive automatically (for example, wide integer bounds), it is acceptable to introduce an `fv_std::assume(condition)` inside the loop. Use assumptions sparingly, and prefer strengthening invariants when possible.

### Summary

- Every `unconstrained fn` may be verified as long as it carries contracts.
- Loops must provide invariants and decreases clauses so the verifier can reason about safety and termination.
- Mutable references are fully interoperable with `fv_std::old()` and can appear in both constrained and unconstrained contexts.

See the `test_programs/formal_verify_success` fixtures (e.g. `verus_requires_ensures`, `verus_assorted_demo`) for concrete, compilable examples of these techniques in action.
