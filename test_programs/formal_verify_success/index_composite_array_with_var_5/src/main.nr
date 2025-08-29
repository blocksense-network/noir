// Here we are testing for a structure which has fields
// of the same type.
struct A { first: u32, second: u32, }
#['requires((x < 2) & (arr[x].first == 4) )]
#['ensures(result == 4)]
fn main(arr: [A;2], x: u32) -> pub u32 {
    arr[x].first
}

