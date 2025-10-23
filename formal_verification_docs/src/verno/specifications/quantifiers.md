# Quantifiers
## Logical implication

To improve readability, Verno supports the *implication* operator `==>`. The expression `a ==> b` (reads as “a implies b”) is logically equivalent to `!a | b`.

For example, the expression:

```
forall|i, j| (0 <= i) & (i <= j) & (j < len) ==> f(i, j)
```

means that for every pair `i` and `j` such that `0 <= i <= j < len`, `f(i, j)` must hold.

Note that `==>` has lower precedence than other Boolean operators. For instance, `a ==> b & c` is interpreted as `a ==> (b & c)`.

## Forall

Suppose we need to specify that all the elements of an array are powers of 2.
If the array is small, we could write a specification for every element separately:

```rust,ignore
fn is_power_of_2(x: i32) -> bool {
  if x <= 0 { 
      false
    } else {
      (x & (x - 1)) == 0
    }
}

#['requires(is_power_of_2(arr[0]))]
#['requires(is_power_of_2(arr[1]))]
#['requires(is_power_of_2(arr[2]))]
#['ensures(is_power_of_2(result))]
fn main(arr: [i32; 3]) -> pub i32 {
    arr[1]
}
```

However, this approach doesn't scale well for larger arrays.

Fortunately, Verno and SMT solvers support the [universal(`forall`) and existential(`exists`) quantifiers](https://en.wikipedia.org/wiki/Quantifier_(logic)), which we can think of as infinite conjunctions or disjunctions:

```
forall(|i| f(i)) = ... f(-2) & f(-1) & f(0) & f(1) & f(2) & ...
exists(|i| f(i)) = ... f(-2) | f(-1) | f(0) | f(1) | f(2) | ...
```
By default the bound variables (`i` in `forall|i|`) are of type `int`, representing **all** mathematical integers, both positive and negative. The SMT solver contains direct support for reasoning about `int` values.

With quantifiers, it's much more convenient to write a specification about all elements of an array:

```rust,ignore
#['requires(
    forall(|i|
     (0 <= i) & (i < 3) ==> is_power_of_2(arr[i])
    ))]
#['ensures(is_power_of_2(result))]
fn main(arr: [i32; 3]) -> pub i32 {
    arr[1]
}
```

## Exists

`exists` expressions are the dual of `forall`. While `forall(|i| f(i))` means that `f(i)` is true for all `i`, `exists(|i| f(i))` asserts that `f(i)` holds for at least one `i`. To prove `exists(|i| f(i))`, an SMT solver has to find one value for `i` such that `f(i)` is true. This value is called a *witness* for `exists(|i| f(i))`.

The following example demonstrates how `exists` successfully finds a witness

```rust,ignore
#['requires(is_power_of_2(arr[0]))] // Potential witness
#['requires(!is_power_of_2(arr[1]))]
#['requires(is_power_of_2(arr[2]))] // Potential witness
#['ensures(exists(|i| (0 <= i) & (i < 3) ==> is_power_of_2(result[i])))] // i is going to be either 0 or 2
fn main(arr: [i32; 3]) -> pub [i32; 3] {
    arr
}
```

This verification succeeds because at least one element (`arr[0]` or `arr[2]`) is a power of 2, allowing `exists` to identify a witness.
