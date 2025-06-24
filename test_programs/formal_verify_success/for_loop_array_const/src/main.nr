#[ensures(result == 10)]
fn main() -> pub u32 {
  let x = [2, 5];
  let y = x[1];
  let mut sum = 0;

  for i in 0..y {
    sum += i
  }
  sum
}
