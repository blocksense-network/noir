struct A { first: i32, second: u32, }
#[requires(x < 2)]
#[ensures(result == arr[x].first)]
fn main(arr: [A;2], x: u32) -> pub i32 {
    arr[x].first
}

