#['requires((x.0 < 10) & (x.2 < 10))]
#['ensures((result.0 == x.0) & (result.2 == x.2 + x.0))]
fn main(x: (u8, Field, u8)) -> pub (u8, Field, u8) {
    (x.0, x.1, x.2 + x.0)
}
