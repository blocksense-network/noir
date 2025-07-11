#['requires((x < 10) & (y < 3))]
fn main(x: u32, y: u8) {
    let _ = lbs(x, y);
    let _ = rbs(x, y);
}

#['requires((x < 10) & (y < 3))]
#['ensures(result == x << y)]
fn lbs(x: u32, y: u8) -> u32 {
    x << y
}

#['requires((x < 10) & (y < 3))]
#['ensures(result == x >> y)]
fn rbs(x: u32, y: u8) -> u32 {
    x >> y
}
