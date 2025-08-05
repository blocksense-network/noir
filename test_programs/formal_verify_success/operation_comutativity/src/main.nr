#['requires(x < 127)]
fn main(x: u8) {
    let _ = left(x);
    let _ = right(x);
}

#['requires(x <= 126)]
#['ensures(result == ((x + 1) * 2))]
fn left(x: u8) -> u8 {
    (x + 1) * 2
}

#['requires(x <= 126)]
#['ensures(result == (2 * (x + 1)))]
fn right(x: u8) -> u8 {
    (x + 1) * 2
}
