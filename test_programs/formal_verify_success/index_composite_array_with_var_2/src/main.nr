struct A { first: i32, second: u32, }
#['requires(x < 2)]
#['ensures(result == arr[x].second)]
fn main(arr: [A;2], x: u32) -> pub u32 {
    arr[x].second
}

