#['requires((0 < x) & (x < 10))]
fn main(x: i64) {
    let _ = integer_i8(x as i8);
    let _ = integer_i16(x as i16);
    let _ = integer_i32(x as i32);
    let _ = integer_i64(x);
    let _ = integer_u8(x as u8);
    let _ = integer_u16(x as u16);
    let _ = integer_u32(x as u32);
    let _ = integer_u64(x as u64);
}

#['requires((0 < x) & (x < 10))]
#['ensures(result == ! x)]
fn integer_i8(x: i8) -> i8 {
    ! x
}

#['requires((0 < x) & (x < 10))]
#['ensures(result == ! x)]
fn integer_i16(x: i16) -> i16 {
    ! x
}

#['requires((0 < x) & (x < 10))]
#['ensures(result == ! x)]
fn integer_i32(x: i32) -> i32 {
    ! x
}

#['requires((0 < x) & (x < 10))]
#['ensures(result == ! x)]
fn integer_i64(x: i64) -> i64 {
    ! x
}

#['requires(x < 10)]
#['ensures(result == ! x)]
fn integer_u8(x: u8) -> u8 {
    ! x
}

#['requires(x < 10)]
#['ensures(result == ! x)]
fn integer_u16(x: u16) -> u16 {
    ! x
}

#['requires(x < 10)]
#['ensures(result == ! x)]
fn integer_u32(x: u32) -> u32 {
    ! x
}

#['requires(x < 10)]
#['ensures(result == ! x)]
fn integer_u64(x: u64) -> u64 {
    ! x
}
