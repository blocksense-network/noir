#[ensures((y ==> result == x) & (!y ==> result == 0))]
fn main(x: u32, y: bool) -> pub u32 {
    if y {
        x
    } else {
        0
    }
}
