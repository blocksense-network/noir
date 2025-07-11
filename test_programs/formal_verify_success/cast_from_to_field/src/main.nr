#['requires((x == 0) | (x == 1))]
fn main(x: Field) -> pub Field {
    ((x as u32) + 1) as Field
}

