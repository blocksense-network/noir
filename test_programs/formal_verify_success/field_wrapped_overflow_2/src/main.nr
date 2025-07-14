// The prime that defines the integer field is
// p = 21888242871839275222246405745257275088548364400416034343698204186575808495617
// Adding 1 to p-1 should result in 0. y is 1 and x is 0 in Prover.toml.
#['requires((x == 9) & (y == 10))]
fn main(x: Field, y: Field) {
    let z: Field = 21888242871839275222246405745257275088548364400416034343698204186575808495616;

    let w = x - y; // w = -1 which should wrap and be equal to z.
    assert(w == z);

    let q = w * -1; // with this multiplication w should wrap and be equal to 1.  
    assert(q == 1);
}

