// The prime that defines the integer field is
// p = 21888242871839275222246405745257275088548364400416034343698204186575808495617
// . Adding 1 to p-1 should result in 0. y is 1 and x is 0 in Prover.toml.
fn main(x: Field, y: Field) {
    let z: Field = 21888242871839275222246405745257275088548364400416034343698204186575808495616;
    assert(z + y == x);
}
