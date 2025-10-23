# Pre- and postconditions

## Preconditions ("requires" annotations)
Let’s start with a simple example. Suppose we want to verify a function `main` that multiplies a number by 4:

```rust,ignore
fn main(x1: i8) -> pub i8 {
    let x2 = x1 + x1;
    x2 + x2 
}
```

If we run `verno formal-verify` to verify the code we will get the following output:

```admonish tip
You can use the shorter alias `verno fv` instead of `verno formal-verify`.
```

```
error: possible arithmetic underflow/overflow
  ┌─ src/main.nr:2:14
  │
2 │     let x2 = x1 + x1;
  │              --------
  │

Error: Verification failed!
```
Verno cannot prove that the result of `x1 + x1` fits in an `i8` value, i.e. is in the range `-128`…`127`. For example, if `x1` were `100`, then `x1 + x1` would be `200`, which exceeds `127`. We need to make sure that the argument `x1` stays within a safe range.

We can do this by adding preconditions (also known as `requires` annotations) to `main` specifying which values for `x1` are allowed. Preconditions are written using Noir's [attributes](https://noir-lang.org/docs/noir/concepts/functions#attributes) syntax and they have a boolean body expression which can utilize the function's arguments:
```rust,ignore
#['requires(-64 <= x1 & x1 < 64)]
fn main(x1: i8) -> pub i8 {
    let x2 = x1 + x1;
    x2 + x2 
}
```
The precondition above says that `x1` must be at least `-64` and less than `64`, so that `x1 + x1` will fit in the range `-128`…`127`. This fixes the error about `x1 + x1`, but we still get an error about `x2 + x2`:
```
error: possible arithmetic underflow/overflow
  ┌─ src/main.nr:4:5
  │  
4 │ ╭     x2 + x2
5 │ │ }
  │ ╰'
  │  

Error: Verification failed!
```
If we want both `x1 + x1` and `x2 + x2` to succeed, we need a stricter bound on `x1`:
```rust,ignore
#['requires(-32 <= x1 & x1 < 32)]
fn main(x1: i8) -> pub i8 {
    let x2 = x1 + x1;
    x2 + x2 
}
```
Now the code verifies successfully!
## Checking Preconditions at Call Sites
Let's rename the function `main` to `quadruple`. Now suppose we try to call `quadruple` with a value that does not satisfy `quadruple`'s precondition.
```rust,ignore
fn main() {
    let n = quadruple(40);
}
```
For this call Verno reports an error, since `40` is not less than `32`:
```
error: precondition not satisfied
  ┌─ src/main.nr:1:12
  │
1 │ #['requires(-32 <= x1 & x1 < 32)]
  │            -------------------- failed precondition
  │

Error: Verification failed!
```
If we pass `25` instead of `40`, verification succeeds: 
```rust,ignore
fn main() {
    let n = quadruple(25);
}
```
## Postconditions ("ensures" annotations)

Postconditions allow us to specify properties about the return value of a function. Let’s revisit the `quadruple` function and verify that its return value is indeed **four times** the input argument. Let's try putting an assertion in `main` to check that `quadruple(10)` returns `40`:
```rust,ignore
fn main() {
    let n = quadruple(10);
    assert(n == 40);
}
```
Although `quadruple(10)` does return `40`, Verno nevertheless reports an error:
```
error: assertion failed
   ┌─ src/main.nr:10:12
   │
10 │     assert(n == 40);
   │            -------- assertion failed
   │

Error: Verification failed!
```
The error occurs because, even though `quadruple` multiplies its argument by `4`, `quadruple` doesn’t publicize this fact to the other functions in the program. To do this, we can add postconditions (`ensures` annotations) to `quadruple` specifying some properties of `quadruple`’s return value. In Verno, the return value is referred with the variable name `result`:
```rust,ignore
#['requires(-32 <= x1 & x1 < 32)]
#['ensures(result == 4 * x1)]
fn quadruple(x1: i8) -> pub i8 {
    let x2 = x1 + x1;
    x2 + x2 
}
```
With this postcondition, Verno can now prove that `quadruple` behaves as intended. When `main` calls quadruple, the SMT solver uses the postcondition to verify the assertion `n == 40`.
### Modular Verification
Preconditions and postconditions enable modular verification, a key feature of Verno. This approach establishes a clear contract between functions:
1. When `main` calls `quadruple`, Verno checks that the arguments satisfy `quadruple`’s preconditions.
2. When verifying `quadruple`’s body, Verno assumes the preconditions hold and ensures the postconditions are satisfied.
3. When verifying `main`, Verno relies on `quadruple`’s postconditions without needing to inspect its implementation.

This modularity breaks verification into smaller, manageable pieces, making it more efficient than verifying the entire program at once. However, writing precise preconditions and postconditions requires effort. For large programs, you’ll likely spend significant time crafting these specifications to ensure correctness.

### Assert and SMT Solvers
During verification with `verno formal-verify`, the `assert` keyword is used to make local requests to the SMT solver (Z3) to prove a specific fact. Importantly, Noir's `assert` behaves differently depending on whether you’re running `verno formal-verify` (verification) or `nargo execute` (execution). 
