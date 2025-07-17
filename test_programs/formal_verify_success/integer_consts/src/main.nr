use fv_std;

#[requires(x <= fv_std::U8_MAX & y <= fv_std::U8_MAX)]
#[ensures((x <= fv_std::U8_MAX - y) ==> result == x + y)]
fn safe_add_u8(x: u8, y: u8) -> u8 {
    if x <= fv_std::U8_MAX - y {
        x + y
    } else {
        0u8
    }
}

fn safe_sub_i16(a: i16, b: i16) -> i16 {
    if (b >= 0 & a >= b) | (b < 0 & a <= fv_std::I16_MAX + b) {
        a - b
    } else {
        0
    }
}

#[ensures(result == (p & q))]
fn bitwise_and_u32(p: u32, q: u32) -> u32 {
    p & q
}

#[requires(p <= fv_std::U32_MAX & q <= fv_std::U32_MAX)]
#[ensures((p <= fv_std::U32_MAX - q) ==> result == p + q)]
fn safe_add_u32(p: u32, q: u32) -> u32 {
    if p <= fv_std::U32_MAX - q {
        p + q
    } else {
        0u32
    }
}

#[requires((x <= fv_std::U8_MAX) & (y <= fv_std::U8_MAX))]
fn main(x: u8, y: u8, a: i16, b: i16, p: u32, q: u32) {
    let _ = safe_add_u8(x, y);
    let _ = safe_sub_i16(a, b);
    let _ = bitwise_and_u32(p, q);
    let _ = safe_add_u32(p, q);
}

