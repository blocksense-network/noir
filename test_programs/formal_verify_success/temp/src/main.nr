fn min(x: u32, y: u32) -> u32 {
    if x < y { x } else { y }
}

#[ensures(result == 2 * min(min(x, y), y))]
#[requires(x < 100 & y < 100)]
fn main(x: u32, y: pub u32) -> pub u32 {
    let z = min(x, y);
    z * 2
}
