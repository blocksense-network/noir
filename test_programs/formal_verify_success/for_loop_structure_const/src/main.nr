struct A {
  a: u32,
  b: u32
}

#[ensures(result == 10)]
fn main() -> pub u32 {
  let x = A{a: 2, b: 5};
  let y = x.b;
  let mut sum = 0;

  for i in 0..y {
    sum += i
  }
  sum
}
