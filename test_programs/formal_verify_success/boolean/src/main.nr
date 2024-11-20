fn main(x: bool, y: bool) {
	let _ = xor(x, y);
	let _ = and(x, y);
	let _ = or(x, y);
	let _ = not(x);
}

#[ensures(result == x ^ y)]
fn xor(x: bool, y: bool) -> bool {
    x ^ y
}

#[ensures(result == x & y)]
fn and(x: bool, y: bool) -> bool {
    x & y
}

#[ensures(result == x | y)]
fn or(x: bool, y: bool) -> bool {
    x | y
}

#[ensures(result == ! x)]
fn not(x: bool) -> bool {
    ! x
}
