// This test shows that you can use the implication operator not
// only in attributes but also in the function's body.
#[requires(y ==> x > 5)]
fn main(x: u32, y: bool) {
    assert(y ==> x > 5);
}
