#[requires((10 < x) & (x < 10) & (0 < y) & (y < 10))]
fn main(x: u8, y: u8) {
    let _ = plus_u8(x, y);
    let _ = minus_u8(x, y);
    let _ = mult_u8(x, y);
    let _ = div_u8(x, y);
    let _ = mod_u8(x, y);
    let _ = xor_u8(x, y);
    let _ = and_u8(x, y);
    let _ = or_u8(x, y);
    let _ = not_u8(x);
}

#[requires((x < 100) & (y < 10))]
#[ensures(result == x + y)]
fn plus_u8(x: u8, y: u8) -> u8 {
    x + y
}

#[requires((10 < x) & (x < 100) & (y < 10))]
#[ensures(result == x - y)]
fn minus_u8(x: u8, y: u8) -> u8 {
    x - y
}

#[requires((x < 10) & (y < 10))]
#[ensures(result == x * y)]
fn mult_u8(x: u8, y: u8) -> u8 {
    x * y
}

#[requires(y != 0)]
#[ensures(result == x / y)]
fn div_u8(x: u8, y: u8) -> u8 {
    x / y
}

#[requires(y != 0)]
#[ensures(result == x % y)]
fn mod_u8(x: u8, y: u8) -> u8 {
    x % y
}

#[ensures(result == x ^ y)]
fn xor_u8(x: u8, y: u8) -> u8 {
    x ^ y
}

#[ensures(result == x & y)]
fn and_u8(x: u8, y: u8) -> u8 {
    x & y
}

#[ensures(result == x | y)]
fn or_u8(x: u8, y: u8) -> u8 {
    x | y
}

#[ensures(result == ! x)]
fn not_u8(x: u8) -> u8 {
    ! x
}
