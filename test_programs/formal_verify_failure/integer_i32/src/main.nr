#[requires((-10 < x) & (x < 10) & (0 < y) & (y < 10))]
fn main(x: i32, y: i32) {
    let _ = plus_i32(x, y);
    let _ = minus_i32(x, y);
    let _ = mult_i32(x, y);
    let _ = div_i32(x, y);
    let _ = mod_i32(x, y);
    let _ = xor_i32(x, y);
    let _ = and_i32(x, y);
    let _ = or_i32(x, y);
    let _ = not_i32(x);
}

#[requires((-10 < x) & (x < 10) & (-10 < y) & (y < 10))]
#[ensures(result == x + y)]
fn plus_i32(x: i32, y: i32) -> i32 {
    x + y
}

#[requires((-10 < x) & (x < 10) & (-10 < y) & (y < 10))]
#[ensures(result == x - y)]
fn minus_i32(x: i32, y: i32) -> i32 {
    x - y
}

#[requires((-10 < x) & (x < 10) & (-10 < y) & (y < 10))]
#[ensures(result == x * y)]
fn mult_i32(x: i32, y: i32) -> i32 {
    x * y
}

#[requires(y != 0)]
#[ensures(result == x / y)]
fn div_i32(x: i32, y: i32) -> i32 {
    x / y
}

#[requires(y != 0)]
#[ensures(result == x % y)]
fn mod_i32(x: i32, y: i32) -> i32 {
    x % y
}

#[ensures(result == x ^ y)]
fn xor_i32(x: i32, y: i32) -> i32 {
    x ^ y
}

#[ensures(result == x & y)]
fn and_i32(x: i32, y: i32) -> i32 {
    x & y
}

#[ensures(result == x | y)]
fn or_i32(x: i32, y: i32) -> i32 {
    x | y
}

#[ensures(result == ! x)]
fn not_i32(x: i32) -> i32 {
    ! x
}
