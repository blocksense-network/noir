fn main(x: u32) -> pub u32 {
    if x > 5 { foo(x) } else { x }
}

#['requires(x > 5)]
fn foo(x: u32) -> u32 {
    0
}
