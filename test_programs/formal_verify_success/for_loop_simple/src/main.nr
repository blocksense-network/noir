#['ensures(result == 10)]
fn main() -> pub u32 {
  let x = 2 + 3;
  let mut sum = 0;
  for i in 0..x {
    sum += i
  }
  sum
}
