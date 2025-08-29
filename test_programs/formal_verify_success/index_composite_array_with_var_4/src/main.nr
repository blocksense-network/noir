struct A {
    first: i32,
    second: u32,
}

#['requires((x < 2) & (arr[x].second == 1))]
#['ensures(result == 1)]
fn main(arr: [A; 2], x: u32) -> pub u32 {
   arr[x].second
}

