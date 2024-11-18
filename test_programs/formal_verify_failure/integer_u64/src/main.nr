#[requires((10 < x) & (x < 100) & (0 < y) & (y < 10))]
fn main(x: u64, y: u64) {
    let _ = plus_u64(x, y);
    let _ = minus_u64(x, y);
    let _ = mult_u64(x, y);
    let _ = div_u64(x, y);
    let _ = mod_u64(x, y);
    let _ = xor_u64(x, y);
    let _ = and_u64(x, y);
    let _ = or_u64(x, y);
    let _ = not_u64(x);
}

#[requires((x < 100) & (y < 10))]
#[ensures(result == x + y)]
fn plus_u64(x: u64, y: u64) -> u64 {
    x + y
}

#[requires((10 < x) & (x < 100) & (y < 10))]
#[ensures(result == x - y)]
fn minus_u64(x: u64, y: u64) -> u64 {
    x - y
}

#[requires((x < 100) & (y < 10))]
#[ensures(result == x * y)]
fn mult_u64(x: u64, y: u64) -> u64 {
    x * y
}

#[requires(y != 0)]
#[ensures(result == x / y)]
fn div_u64(x: u64, y: u64) -> u64 {
    x / y
}

#[requires(y != 0)]
#[ensures(result == x % y)]
fn mod_u64(x: u64, y: u64) -> u64 {
    x % y
}

#[ensures(result == x ^ y)]
fn xor_u64(x: u64, y: u64) -> u64 {
    x ^ y
}

#[ensures(result == x & y)]
fn and_u64(x: u64, y: u64) -> u64 {
    x & y
}

#[ensures(result == x | y)]
fn or_u64(x: u64, y: u64) -> u64 {
    x | y
}

#[ensures(result == ! x)]
fn not_u64(x: u64) -> u64 {
    ! x
}
