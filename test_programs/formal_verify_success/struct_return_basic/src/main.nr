struct A {
    x: u32,
    y: u32,
}

#['ensures(result == 3)]
fn main() -> pub u32 {
    let A { x, y } = A {x: 1, y: 2};
    x + y
}
