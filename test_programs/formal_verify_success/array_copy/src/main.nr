// ArrayGet is not yet implemented
#['ensures((result[0] == x[0]) & (result[1] == 1) & (result[2] == x[2]))]
fn main(x: [u8; 3]) -> pub [u8; 3] {
	[x[0], 1, x[2]]
}
