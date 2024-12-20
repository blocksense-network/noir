// This test is expected to fail because if y is true, main returns x, not 0.
#[ensures((y ==> result == 0))]
fn main(x: u32, y: bool) -> pub u32 {
    if y {
        x
    } else {
        0
    }
}
