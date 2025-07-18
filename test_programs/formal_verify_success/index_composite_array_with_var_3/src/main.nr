struct A {
    first: i32,
    second: u32,
    third: (u32, u32),
}
struct C {
    a: u32,
    b: [A; 32],
}
struct B {
    x: [[A; 192]; 74],
    y: ([A;19],(i16, bool, (i8, [(C, u32);5], u32), u16))
}

#['requires((x < 2) & (arr[x].y.1.2.1[x].0.b[x].third.0 == 1))]
#['ensures(result == arr[x].y.1.2.1[x].0.b[x].third.0 / 2)]
fn main(arr: [B;2], x: u32) -> pub u32 {
    arr[x].y.1.2.1[x].0.b[x].third.0 / 2
}

