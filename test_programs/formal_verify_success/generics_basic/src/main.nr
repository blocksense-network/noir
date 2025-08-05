#['requires(generic_func(x, x))]
fn main(x: u32) {
}

#['ghost]
fn generic_func<T>(x: T, y: u32) -> bool {
  true
}
