#[ensures(result == 10)]
fn main() -> pub u32 {
let x = 3;
let mut sum = 0;
  {
    let x = 5;
    for i in 0..x {
        // iterates 5 times, not 3 times
        sum += i;
    }
  }
sum
}
