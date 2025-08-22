// Verification should pass because all overflows
// are prevented by the if condition.
// We also expect the requires clause 
// to work correctly and make the test
// equivalent to if_test_overflow_1
#['requires(y == 255)]
fn main(x: u8, y: u8) -> pub u8 {
    if x < y {
        x + 1
    }
    else {
        0
    }
}
