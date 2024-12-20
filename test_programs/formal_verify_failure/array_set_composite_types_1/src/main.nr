fn main(mut x: [(u32, i32); 5], y: i32, index: u32) -> pub [(u32, i32); 5] {
    x[index].1 = y;
    x
}

