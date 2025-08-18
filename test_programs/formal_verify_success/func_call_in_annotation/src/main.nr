#['ensures(foo(5))]
fn main() {}

#['ghost]
fn foo(x: u16) -> bool {
  true
}
