#[requires((x % 2 == 0) & (z % 2 == 0)
                        & (x < z )
                        & (z < 1000))]
#[ensures(result == 0)]
fn main(x: u32, z: u32) -> pub u32 {
    let mut y = x + z;
     y % 2
}

