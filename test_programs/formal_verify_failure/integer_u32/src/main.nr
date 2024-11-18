#[requires((10 < x) & (x < 100) & (0 < y) & (y < 10))]
fn main(x: u32, y: u32) {
    let _ = plus_u32(x, y);
    let _ = minus_u32(x, y);
    let _ = mult_u32(x, y);
    let _ = div_u32(x, y);
    let _ = mod_u32(x, y);
    let _ = xor_u32(x, y);
    let _ = and_u32(x, y);
    let _ = or_u32(x, y);
    let _ = not_u32(x);
}

#[requires((x < 100) & (y < 10))]
#[ensures(result == x + y)]
fn plus_u32(x: u32, y: u32) -> u32 {
    x + y
}

#[requires((10 < x) & (x < 100) & (y < 10))]
#[ensures(result == x - y)]
fn minus_u32(x: u32, y: u32) -> u32 {
    x - y
}

#[requires((x < 100) & (y < 10))]
#[ensures(result == x * y)]
fn mult_u32(x: u32, y: u32) -> u32 {
    x * y
}

#[requires(y != 0)]
#[ensures(result == x / y)]
fn div_u32(x: u32, y: u32) -> u32 {
    x / y
}

#[requires(y != 0)]
#[ensures(result == x % y)]
fn mod_u32(x: u32, y: u32) -> u32 {
    x % y
}

#[ensures(result == x ^ y)]
fn xor_u32(x: u32, y: u32) -> u32 {
    x ^ y
}

#[ensures(result == x & y)]
fn and_u32(x: u32, y: u32) -> u32 {
    x & y
}

#[ensures(result == x | y)]
fn or_u32(x: u32, y: u32) -> u32 {
    x | y
}

#[ensures(result == ! x)]
fn not_u32(x: u32) -> u32 {
    ! x
}
