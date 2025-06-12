fn main(x: u32) -> pub [u32; 3] {
    if x > 5 { foo(x) } else { [x, x, x] }
}

#[requires(x > 5)]
fn foo(x: u32) -> [u32; 3] {
    [0, 0, 0]
}
