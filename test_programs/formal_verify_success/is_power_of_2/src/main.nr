#[ghost]
fn is_power_of_2(x: i32) -> bool {
  if x <= 0 {
      false
    } else {
      (x & (x - 1)) == 0
    }
}

#[requires(is_power_of_2(arr[0]))]
#[ensures(is_power_of_2(result))]
fn main(arr: [i32; 3]) -> pub i32 {
    arr[0]
}

