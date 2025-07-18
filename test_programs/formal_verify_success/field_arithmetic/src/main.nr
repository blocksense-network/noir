// Fields wrap around when overflowing. Therefore this program
// should not produce a "potential overflow" error.
fn main(x: Field, y: Field) {
    let _ = plus(x, y);
    let _ = minus(x, y);
    let _ = mult(x, y);
    let _ = div(x, y);
}

#['ensures(result == x + y)]
fn plus(x: Field, y: Field) -> Field {
    x + y
}

#['ensures(result == x - y)]
fn minus(x: Field, y: Field) -> Field {
    x - y
}

#['ensures(result == x * y)]
fn mult(x: Field, y: Field) -> Field {
    x * y
}

#['ensures(result == x / y)]
fn div(x: Field, y: Field) -> Field {
    x / y
}
