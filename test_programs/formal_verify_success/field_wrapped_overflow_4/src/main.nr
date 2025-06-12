// The prime that defines the integer field is
// p = 21888242871839275222246405745257275088548364400416034343698204186575808495617
// Adding p-1 + p-1 + p-1 should result in p-3.
// Goal is to prove (p - 3 + 3) == 0
#[requires((x == 0) & (y == 21888242871839275222246405745257275088548364400416034343698204186575808495616))]
fn main(x: Field, y: Field) {
    let z = y + y + y;
    assert(z + 3 == x);
}

