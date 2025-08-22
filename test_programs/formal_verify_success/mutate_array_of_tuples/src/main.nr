#['requires(x[3].0 == -2)]
#['requires(i < 5)]
#['ensures(result[i].1 == y)]
fn main(mut x: [(i32, u32); 5], i: u32, y: u32) -> pub [(i32, u32); 5] {
    x[i].1 = y;
    x
}
