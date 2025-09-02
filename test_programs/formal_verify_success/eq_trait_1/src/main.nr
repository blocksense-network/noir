#['ensures(result == x)]
fn main(x: u32) -> pub u32 {
	id(x)
}

#['ensures(result == x)]
fn id<T>(x: T) -> T
 where T: Eq {
  if x == x {
    x
  } else {
    x
  }
}
