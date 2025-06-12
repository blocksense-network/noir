// The prime that defines the integer field is
// p = 21888242871839275222246405745257275088548364400416034343698204186575808495617
// If we have the value (p+1)/2 and we multiply it by 2 we will get p+1
// but if we divide p+1 by 2 we will not get the original value but we will get 0.
#[requires((x == 2) & (y == 0))]
fn main(x: Field, y: Field) {
    let z: Field = 10944121435919637611123202872628637544274182200208017171849102093287904247809; // ((p + 1) / 2)
    let w = z * x; // ((p + 1)/2) * 2  ==> w == p + 1 == 1
    assert(w / 2 != z); // w / 2 != (p + 1) / 2
    assert(w / 2 == y); // w / 2 == 0
}

