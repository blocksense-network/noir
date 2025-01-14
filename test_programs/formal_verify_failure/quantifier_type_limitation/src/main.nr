#[requires(forall(|i| (x < i) & (i as u64 < y)))]
fn main(x: u32, y: u64) {
  // This tests shows how you must cast the quantifier index
  // into a specific type. This only occurs if you are using the index
  // with variables of a different type.
}
