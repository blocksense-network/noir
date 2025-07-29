#['ensures(result == 10)]
fn main() -> pub u32 {
  let x = {
    let a = 3;
    a + {1 + 1}
  };
  let mut sum = 0;
  for i in 0..x {
    sum += i
  }
  sum
}
