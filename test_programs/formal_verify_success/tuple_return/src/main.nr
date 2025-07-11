#['requires(x < 1000)]
#['ensures((result.0 == x + 1) & (result.1 == x + 2) & (result.2 == x + 3))]
fn f(x: u32) -> (u32, u32, u32) {
    (x + 1, x + 2, x + 3)
}

#['requires(x < 1000)]
#['ensures((result.0 == x + 3) & (result.1 == x + 1) & (result.2 == x + 2))]
fn g(x: u32) -> (u32, u32, u32) {
    let res = f(x);
    (res.2, res.0, res.1)
}

#['ensures(result == 7)]
fn main() -> pub u32 {
    let (a, b, c) = g(5);
    a + b - c
}
