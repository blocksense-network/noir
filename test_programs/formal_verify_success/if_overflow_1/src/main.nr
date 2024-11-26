// Verification should pass because all overflows
// are prevented by the if condition
fn main(x: u8) -> pub u8 {
    if x < 255 {
        x + 1
    }
    else {
        0
    }
}
