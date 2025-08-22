struct A { x: u8, y: u8 }

#['requires((a.x < 10) & (a.y < 10))]
#['ensures((result.x == a.x + a.y) & (result.y == a.y))]
fn main(a: A) -> pub A {
	A { x: a.x + a.y, y: a.y }
}
