#['requires(forall(|i| (0 <= i) & (i < 10) ==> arr[i] % 2 == 0) 
            & forall(|i| (0 <= i) & (i < 10) ==> arr[i] < 100) )]
#['ensures((result % 2 == 0) & (result < 1000))]
// The sum of an array which constist of even elements upper bounded by 100
// will be even and upper bounded by 1000
fn main(arr: [u32; 10]) -> pub u32 {
    let mut sum = 0;
    for i in 0..10 {
        sum += arr[i];
    }
    sum
}
