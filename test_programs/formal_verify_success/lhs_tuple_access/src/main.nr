#['ensures(result.1 == x)]
fn main(x: u8) -> pub (u8, u8) {
  let mut y = (3, 5);
  y.1 = x;
  y
}

