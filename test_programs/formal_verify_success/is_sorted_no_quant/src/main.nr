fn is_ascending_sorted(arr: [u64; 3]) -> bool {
    let mut is_sorted: bool = true;
    for i in 0..2 {
        if arr[i] > arr[i + 1] {
            is_sorted = false;
        }
    }
    is_sorted
}
#[requires(arr[0] <= arr[1])]
#[requires(arr[1] <= arr[2])]
#[ensures(is_ascending_sorted(arr))]
fn main(arr: [u64; 3])
{
 //   assert(is_ascending_sorted(arr));
}
