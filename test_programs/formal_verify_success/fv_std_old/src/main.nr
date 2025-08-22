use fv_std::old;

#['requires(x == 4)]
#['ensures(result == 5)]
fn main(mut x: u32) -> pub u32 {
  foo(&mut x);
  x
}

#['requires(*old(x) == 4)]
#['ensures(*x == *old(x) + 1)]
fn foo(x: &mut u32) {
  *x = *x + 1;
}


