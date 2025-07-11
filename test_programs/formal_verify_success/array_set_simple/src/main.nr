#['ensures(result == 15)]
fn main(x: [u32; 5], i: u32, y: u32) -> pub u32 {
    let mut arr = x;
    arr[3] = 15;
    arr[3]
}

