#[foreign(schnorr_verify)]
// docs:start:schnorr_verify
pub fn verify_signature<let N: u32>(
    public_key_x: Field,
    public_key_y: Field,
    signature: [u8; 64],
    message: [u8; N]
) -> bool
// docs:end:schnorr_verify
{}

#[foreign(schnorr_verify)]
// docs:start:schnorr_verify_slice
pub fn verify_signature_slice(
    public_key_x: Field,
    public_key_y: Field,
    signature: [u8; 64],
    message: [u8]
) -> bool
// docs:end:schnorr_verify_slice
{}

