// This test is expected to fail
#[requires(y != 255)]
fn main(x: u8, y: u16) -> pub u8 {
    if x as u16 < y {
        x + 1
    }
    else {
        0
    }
}
