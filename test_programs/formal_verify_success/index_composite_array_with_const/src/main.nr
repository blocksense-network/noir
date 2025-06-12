struct A { first: i32, second: u32, }
#[ensures(result == arr[1].second)]
fn main(arr: [A;2]) -> pub u32 {
    arr[1].second
}
