#[requires(x <= 252)]
// Instruction indexing causes a crash when the InstructionId is inside fv_instructions in the dfg
fn main(x: u8) {
    let _ = left(x);
    let _ = right(x);
}

#[ensures(result == ((x + 1) + 2))]
fn left(x: u8) -> u8 {
    (x + 1) + 2
}

#[ensures(result == (2 + (x + 1)))]
fn right(x: u8) -> u8 {
    2 + (x + 1)
}
