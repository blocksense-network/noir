#[requires((-10 < x) & (x < 10) & (0 < y) & (y < 10))]
fn main(x: i8, y: i8) {
    let _ = plus_i8(x, y);
    let _ = minus_i8(x, y);
    let _ = mult_i8(x, y);
    let _ = div_i8(x, y);
    let _ = mod_i8(x, y);
    let _ = xor_i8(x, y);
    let _ = and_i8(x, y);
    let _ = or_i8(x, y);
    let _ = not_i8(x);
}

#[requires((-10 < x) & (x < 10) & (-10 < y) & (y < 10))]
#[ensures(result == x + y)]
fn plus_i8(x: i8, y: i8) -> i8 {
    x + y
}

#[requires((-10 < x) & (x < 10) & (-10 < y) & (y < 10))]
#[ensures(result == x - y)]
fn minus_i8(x: i8, y: i8) -> i8 {
    x - y
}

#[requires((-10 < x) & (x < 10) & (-10 < y) & (y < 10))]
#[ensures(result == x * y)]
fn mult_i8(x: i8, y: i8) -> i8 {
    x * y
}

#[requires(y != 0)]
#[ensures(result == x / y)]
fn div_i8(x: i8, y: i8) -> i8 {
    x / y
}

#[requires(y != 0)]
#[ensures(result == x % y)]
fn mod_i8(x: i8, y: i8) -> i8 {
    x % y
}

#[ensures(result == x ^ y)]
fn xor_i8(x: i8, y: i8) -> i8 {
    x ^ y
}

#[ensures(result == x & y)]
fn and_i8(x: i8, y: i8) -> i8 {
    x & y
}

#[ensures(result == x | y)]
fn or_i8(x: i8, y: i8) -> i8 {
    x | y
}

#[ensures(result == ! x)]
fn not_i8(x: i8) -> i8 {
    ! x
}
