#['requires(x < 1000)]
#['ensures((result.0 == x + 1) & (result.1 == x + 2) & (result.2 == x + 3))]
fn f(x: u32) -> (u32, u32, u32) {
    (x + 1, x + 2, x + 3)
}

#['ensures(result == 5)]
fn main() -> pub u32 {
    let (a, b, c) = f(5);
    a + b - c
}
