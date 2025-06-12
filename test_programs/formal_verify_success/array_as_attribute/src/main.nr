fn main() {
    let _ = foo1([1, 1, 8, 0, 0]);
    let _ = foo2([8, 1, 8, 0, 0]);
    let _ = foo3([1, 8, 8, 0, 0]);
    let _ = foo4([1, 0, 8, 0, 0]);
}

#[requires((x[1] == 1) & (x[0] == 1) & (x[2] == 8))]
fn foo1(x: [u32; 5]) {
   assert(x[2] == 8);
}

#[requires((x[1] == 1) & (x[0] == 8) & (x[2] == 8))]
fn foo2(x: [u32; 5]) {
   assert(x[2] == 8);
}

#[requires((x[1] == 8) & (x[0] == 1) & (x[2] == 8))]
fn foo3(x: [u32; 5]) {
   assert(x[2] == 8);
}

#[requires((x[0] == 1) & (x[2] == 8))]
fn foo4(x: [u32; 5]) {
   assert(x[2] == 8);
}
