#[requires(x == "ab")]
#[ensures(result[0] == x[0])]
fn main(x: str<2>) -> pub str<2> {
    x
}
