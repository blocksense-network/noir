# Pre- and postconditions

## Preconditions (requires attributes)
Let’s start with a simple example. Suppose we want to verify a function `main` that multiplies a number by 4:

```rust,ignore
fn main(x1: i8) -> pub i8 {
    let x2 = x1 + x1;
    x2 + x2 
}
```

If we run `nargo fv` to verify the code we will get the following output:
```
error: possible arithmetic underflow/overflow
  ┌─ src/main.nr:2:14
  │
2 │     let x2 = x1 + x1;
  │              --------
  │

Error: Verification failed!
```
Noir FV cannot prove that the result of `x1 + x1` fits in an 8-bit `i8` value, which allows values in the range `-128`…`127`. For example, if `x1` were `100`, then `x1 + x1` would be `200`, which exceeds `127`. We need to make sure that the argument `x1` stays within a safe range.  
We can do this by adding preconditions (also known as `requires` attributes) to `main` specifying which values for `x1` are allowed. In Noir FV, preconditions are written using Noir's attributes syntax:
```rust,ignore
#[requires(-64 <= x1 & x1 < 64)]
fn main(x1: i8) -> pub i8 {
    let x2 = x1 + x1;
    x2 + x2 
}
```
The two preconditions above say that x1 must be at least `-64` and less than `64`, so that `x1 + x1` will fit in the range `-128`…`127`. This fixes the error about `x1 + x1`, but we still get an error about `x2 + x2`:
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
#[requires(-32 <= x1 & x1 < 32)]
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
For this call Noir FV reports an error, since 40 is not less than 32:
```
error: precondition not satisfied
  ┌─ src/main.nr:1:12
  │
1 │ #[requires(-32 <= x1 & x1 < 32)]
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
