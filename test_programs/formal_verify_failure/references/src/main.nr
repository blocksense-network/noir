#[requires(*x == 0)]
fn set_to_zero(x: &mut Field) {
    *x = 0;
}

#[ensures(result == 0)]
fn main() -> pub Field {
    let mut y = 42;
    set_to_zero(&mut y);
    y
}
