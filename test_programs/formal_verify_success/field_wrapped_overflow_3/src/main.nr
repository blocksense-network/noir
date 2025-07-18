// The prime that defines the integer field is
// p = 21888242871839275222246405745257275088548364400416034343698204186575808495617
// Adding 1 to p-1 should result in 0.
#['requires((x == 21888242871839275222246405745257275088548364400416034343698204186575808495616) & (y == 1))]
#['ensures(result == x + y)] // Tests if Field sum in annotation is also correct.
fn main(x: Field, y: Field) -> pub Field {
    x + y // Wraps around and the result should equal to zero.
}

