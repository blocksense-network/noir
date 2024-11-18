#[ensures(result == (x as u32) as u8)]
fn main(x: u16) -> pub u8 {
	(x as u32) as u8
}
