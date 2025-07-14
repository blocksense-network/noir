fn main(x: i64, y: i64) {
    let _ = integer_i8(x as i8, y as i8);
    let _ = integer_i16(x as i16, y as i16);
    let _ = integer_i32(x as i32, y as i32);
    let _ = integer_i64(x, y);
    let _ = integer_u8(x as u8, y as u8);
    let _ = integer_u16(x as u16, y as u16);
    let _ = integer_u32(x as u32, y as u32);
    let _ = integer_u64(x as u64, y as u64);
}

#['ensures(result == x & y)]
fn integer_i8(x: i8, y: i8) -> i8 {
    x & y
}

#['ensures(result == x & y)]
fn integer_i16(x: i16, y: i16) -> i16 {
    x & y
}

#['ensures(result == x & y)]
fn integer_i32(x: i32, y: i32) -> i32 {
    x & y
}

#['ensures(result == x & y)]
fn integer_i64(x: i64, y: i64) -> i64 {
    x & y
}

#['ensures(result == x & y)]
fn integer_u8(x: u8, y: u8) -> u8 {
    x & y
}

#['ensures(result == x & y)]
fn integer_u16(x: u16, y: u16) -> u16 {
    x & y
}

#['ensures(result == x & y)]
fn integer_u32(x: u32, y: u32) -> u32 {
    x & y
}

#['ensures(result == x & y)]
fn integer_u64(x: u64, y: u64) -> u64 {
    x & y
}
