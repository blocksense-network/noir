struct A {
    x: u32,
    y: u32,
}

#[requires(x < 1000)]
#[ensures((result.x == x + 1) & (result.y == x + 2))]
fn f(x: u32) -> A {
    A { x: x + 1, y: x + 2 }
}

#[requires(x < 1000)]
#[ensures((result.x == x + 2) & (result.y == x + 1))]
fn g(x: u32) -> A {
    let res = f(x);
    A { x: res.y, y: res.x }
}

#[ensures(result == 13)]
fn main() -> pub u32 {
    let A { x, y } = g(5);
    x + y
}
