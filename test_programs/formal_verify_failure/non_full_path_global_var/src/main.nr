use fv_std::U8_MIN;

// This test will fail because we only support full paths for globals
// from foreign dependencies.
#['ensures(U8_MIN == result )]
fn main() -> pub u8 {
  0
}
