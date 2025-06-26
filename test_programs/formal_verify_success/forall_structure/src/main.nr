struct A {
  id: Field,
  balance: u32,
}
#[requires(forall(|i, j| (0 <= i) & (i < j) & (j < 2) 
                ==> x[i].id != x[j].id))]
fn main(x: [A; 2]) {}

