#[requires((-10 < x) & (x < 10) & (0 < y) & (y < 10))]
fn main(x: i16, y: i16) {
    let _ = plus_i16(x, y);
    let _ = minus_i16(x, y);
    let _ = mult_i16(x, y);
    let _ = div_i16(x, y);
    let _ = mod_i16(x, y);
    let _ = xor_i16(x, y);
    let _ = and_i16(x, y);
    let _ = or_i16(x, y);
    let _ = not_i16(x);
}

#[requires((-10 < x) & (x < 10) & (-10 < y) & (y < 10))]
#[ensures(result == x + y)]
fn plus_i16(x: i16, y: i16) -> i16 {
    x + y
}

#[requires((-10 < x) & (x < 10) & (-10 < y) & (y < 10))]
#[ensures(result == x - y)]
fn minus_i16(x: i16, y: i16) -> i16 {
    x - y
}

#[requires((-10 < x) & (x < 10) & (-10 < y) & (y < 10))]
#[ensures(result == x * y)]
fn mult_i16(x: i16, y: i16) -> i16 {
    x * y
}

#[requires(y != 0)]
#[ensures(result == x / y)]
fn div_i16(x: i16, y: i16) -> i16 {
    x / y
}

#[requires(y != 0)]
#[ensures(result == x % y)]
fn mod_i16(x: i16, y: i16) -> i16 {
    x % y
}

#[ensures(result == x ^ y)]
fn xor_i16(x: i16, y: i16) -> i16 {
    x ^ y
}

#[ensures(result == x & y)]
fn and_i16(x: i16, y: i16) -> i16 {
    x & y
}

#[ensures(result == x | y)]
fn or_i16(x: i16, y: i16) -> i16 {
    x | y
}

#[ensures(result == ! x)]
fn not_i16(x: i16) -> i16 {
    ! x
}
