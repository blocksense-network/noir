#[ensures( (!(x < y) | (result == x)) & (!(x >= y) | (result==y))  )]
fn min(x: u32, y: u32) -> u32 {
    min2(x, y)
}

#[ensures( (!(x < y) | (result == x)) & (!(x >= y) | (result==y))  )]
fn min2(x: u32, y: u32) -> u32 {
    if x < y { x } else { y }
}

#[requires((x < 100) & (y < 100))]
#[ensures(result == 2 * min(min(x, y), y))]
fn main(x: u32, y: pub u32) -> pub u32 {
    let z = min(x, y);
    z * 2
}
