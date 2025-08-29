struct A { x: u8, y: u8, w: B }

struct B { z: u32 }

#['requires((a.x < 10) & (a.y < 10))]
#['ensures((result.x == a.x + a.y) & (result.y == a.y))]
fn main(a: A) -> pub A {
	A { x: a.x + a.y, y: a.y, w: B {z: 5} }
}
