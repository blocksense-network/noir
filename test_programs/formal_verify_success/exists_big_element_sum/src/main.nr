#[requires(exists(|i| (0 <= i) & (i < 20) & arr[i] > 100))]
#[ensures(result > 100)] 
// We require that an element bigger than 100 exists in the array.
// Therefore we expect the sum to be bigger than 100.
fn main(arr: [u16; 20]) -> pub u64 {
    let mut sum: u64 = 0;
    for i in 0..20 {
        sum += arr[i] as u64;
    }
    sum
}
