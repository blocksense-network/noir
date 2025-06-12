#[requires(i < 5)]
#[ensures(result[i] == y)]
fn main(mut x: [u32; 5], i: u32, y: u32) -> pub [u32; 5] {
    x[i] = y;
    x
}

