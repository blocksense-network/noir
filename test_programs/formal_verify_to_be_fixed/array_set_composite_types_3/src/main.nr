struct A {
    first: i32,
    second: (u32, u32, u32),
}

#['requires(x < 1)]
#['ensures(result[x + 1].second.2 == 1)]
fn main(mut arr: [A; 2], x: u32) -> pub [A; 2] {
   arr[x + 1].second.2 = 1;
   arr
}

