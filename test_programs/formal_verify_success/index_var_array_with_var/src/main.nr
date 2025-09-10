#['requires(x == 2)]
#['ensures(result == 65)]
fn main(x: u32) -> pub u32 {
    let arr = [857, 1, 65];
    arr[x]
}
