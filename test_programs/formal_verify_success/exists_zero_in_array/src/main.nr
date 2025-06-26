#[requires(exists(|i| (0 <= i) & (i < 7) & (arr[i] == 0)))]
#[ensures(result == 0)]
fn main(arr: [u8; 7]) -> pub u64 {
    let mut mul: u64 = 1;
    for i in 0..7 {
        mul *= arr[i] as u64;
    }
    mul
}
