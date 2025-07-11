fn main(x: u8, y: u8) {
    let _ = lt(x, y);
    let _ = lte(x, y);
    let _ = gt(y, x);
    let _ = gte(y, x);
    let _ = eq(x, x);
    let _ = neq(x, y);
}

#['ensures(result == (x < y))]
fn lt(x: u8, y: u8) -> bool {
    x < y
}

#['ensures(result == (x <= y))]
fn lte(x: u8, y: u8) -> bool {
    x <= y
}

#['ensures(result == (x > y))]
fn gt(x: u8, y: u8) -> bool {
    x > y
}

#['ensures(result == (x >= y))]
fn gte(x: u8, y: u8) -> bool {
    x >= y
}

#['ensures(result == (x == y))]
fn eq(x: u8, y: u8) -> bool {
    x == y
}

#['ensures(result == (x != y))]
fn neq(x: u8, y: u8) -> bool {
    x != y
}
