#[ensures((result[0] == x) & (result[1] == y))]
fn main(x: u32, y: u32) -> pub [u32; 2] {
    [x, y]
}
