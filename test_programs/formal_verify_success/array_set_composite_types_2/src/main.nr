#[requires(index < 5)]
#[ensures(result[index] == y)]
fn main(mut x: [(u32, i32); 5], y: (u32, i32), index: u32) -> pub [(u32, i32); 5] {
    let _ = x[index] == y;
    x[index] = y;
    x
}

