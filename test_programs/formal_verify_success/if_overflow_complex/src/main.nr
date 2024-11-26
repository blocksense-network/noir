// This test is slightly more complex
// because it conatins nested ifs
fn main(x: u8) -> pub u8 {
    if x >= 255 {
        0
    }
    else {
        if x > 12 {
          x - 1
        } else {
          x + 1
        }
    }
}
