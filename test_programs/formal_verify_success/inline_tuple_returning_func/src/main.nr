#[ensures(result == foo().1 )]
fn main(x: Field, y: pub Field) -> pub u32 {
    foo().1
}

#[ensures(result.1 == 2)]
fn foo() -> (u32, u32){
    (1, 2)
}
