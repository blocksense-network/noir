#[requires((10 < x) & (x < 100) & (0 < y) & (y < 10))]
fn main(x: u16, y: u16) {
    let _ = plus_u16(x, y);
    let _ = minus_u16(x, y);
    let _ = mult_u16(x, y);
    let _ = div_u16(x, y);
    let _ = mod_u16(x, y);
    let _ = xor_u16(x, y);
    let _ = and_u16(x, y);
    let _ = or_u16(x, y);
    let _ = not_u16(x);
}

#[requires((x < 100) & (y < 10))]
#[ensures(result == x + y)]
fn plus_u16(x: u16, y: u16) -> u16 {
    x + y
}

#[requires((10 < x) & (x < 100) & (y < 10))]
#[ensures(result == x - y)]
fn minus_u16(x: u16, y: u16) -> u16 {
    x - y
}

#[requires((x < 100) & (y < 10))]
#[ensures(result == x * y)]
fn mult_u16(x: u16, y: u16) -> u16 {
    x * y
}

#[requires(y != 0)]
#[ensures(result == x / y)]
fn div_u16(x: u16, y: u16) -> u16 {
    x / y
}

#[requires(y != 0)]
#[ensures(result == x % y)]
fn mod_u16(x: u16, y: u16) -> u16 {
    x % y
}

#[ensures(result == x ^ y)]
fn xor_u16(x: u16, y: u16) -> u16 {
    x ^ y
}

#[ensures(result == x & y)]
fn and_u16(x: u16, y: u16) -> u16 {
    x & y
}

#[ensures(result == x | y)]
fn or_u16(x: u16, y: u16) -> u16 {
    x | y
}

#[ensures(result == ! x)]
fn not_u16(x: u16) -> u16 {
    ! x
}
