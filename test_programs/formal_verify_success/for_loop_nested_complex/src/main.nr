#[ensures(result == 2)]
fn main() -> pub u32 {
let x = 2;
let y = 3;
let mut sum = 0;
for i in 0..x {
  for j in i..y {
        sum += i;
    }
 } 
sum
}
