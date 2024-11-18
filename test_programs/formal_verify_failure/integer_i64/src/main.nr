#[requires((-10 < x) & (x < 10) & (0 < y) & (y < 10))]
fn main(x: i64, y: i64) {
    let _ = plus_i64(x, y);
    let _ = minus_i64(x, y);
    let _ = mult_i64(x, y);
    let _ = div_i64(x, y);
    let _ = mod_i64(x, y);
    let _ = xor_i64(x, y);
    let _ = and_i64(x, y);
    let _ = or_i64(x, y);
    let _ = not_i64(x);
}

#[requires((-10 < x) & (x < 10) & (-10 < y) & (y < 10))]
#[ensures(result == x + y)]
fn plus_i64(x: i64, y: i64) -> i64 {
    x + y
}

#[requires((-10 < x) & (x < 10) & (-10 < y) & (y < 10))]
#[ensures(result == x - y)]
fn minus_i64(x: i64, y: i64) -> i64 {
    x - y
}

#[requires((-10 < x) & (x < 10) & (-10 < y) & (y < 10))]
#[ensures(result == x * y)]
fn mult_i64(x: i64, y: i64) -> i64 {
    x * y
}

#[requires(y != 0)]
#[ensures(result == x / y)]
fn div_i64(x: i64, y: i64) -> i64 {
    x / y
}

#[requires(y != 0)]
#[ensures(result == x % y)]
fn mod_i64(x: i64, y: i64) -> i64 {
    x % y
}

#[ensures(result == x ^ y)]
fn xor_i64(x: i64, y: i64) -> i64 {
    x ^ y
}

#[ensures(result == x & y)]
fn and_i64(x: i64, y: i64) -> i64 {
    x & y
}

#[ensures(result == x | y)]
fn or_i64(x: i64, y: i64) -> i64 {
    x | y
}

#[ensures(result == ! x)]
fn not_i64(x: i64) -> i64 {
    ! x
}
