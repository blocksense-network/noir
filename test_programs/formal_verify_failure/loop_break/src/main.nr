// This test fails currently because we are not supporting
// decrease clauses. Otherwise it would pass.
unconstrained fn main(x: u32) -> pub u32 {
    let mut y = x;
    loop {
        y += 0;
        break;
    }
    y
}
