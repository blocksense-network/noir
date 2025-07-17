pub fn old<T>(x: T) -> T {
  x
}

pub fn assume(_x: bool) {}

pub global U8_MIN: u8 = 0;
pub global U8_MAX: u8 = 255;

pub global I8_MIN: i8 = -128;
pub global I8_MAX: i8 = 127;

pub global U16_MIN: u16 = 0;
pub global U16_MAX: u16 = 65535;

pub global I16_MIN: i16 = -32768;
pub global I16_MAX: i16 = 32767;

pub global U32_MIN: u32 = 0;
pub global U32_MAX: u32 = 4294967295;

pub global I32_MIN: i32 = -2147483648;
pub global I32_MAX: i32 = 2147483647;

pub global U64_MIN: u64 = 0;
pub global U64_MAX: u64 = 18446744073709551615;

pub global I64_MIN: i64 = -9223372036854775808;
pub global I64_MAX: i64 = 9223372036854775807;
