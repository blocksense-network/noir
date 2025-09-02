#['ensures((result[0] == x[0]) & (result[1] == x[1]))]
fn main(x: [u32; 2]) -> pub [u32; 2] {
    [x[0], x[1]]
}
