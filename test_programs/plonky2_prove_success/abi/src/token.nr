global ADDRESS_CHARACTERS: Field = 42;

struct Token<let N: u8> {
    tag: Field,
    address: Address,
    integer: Integer,
    bytes: Bytes,
    string: String<N>,
    array: Array,
    fixed_array: FixedArray<N>,
    fixed_bytes: FixedBytes<N>,
    is_bool: bool,
    // tuple: Tuple,
}

// Address
struct Address {
    value: str<ADDRESS_CHARACTERS>
}

// Int(size),
// Uint(size),
struct Integer {
    value: Field,
    size: Field
}

// Bytes
struct Bytes {
    value: [u8]
}

// String
struct String<let N: u8> {
    value: str<N>
}

// Array(inner),
struct Array {
    value: [Field],
}

// FixedArray(inner, size),
struct FixedArray<let N: u8> {
    value: [Field; N],
}

// FixedBytes
struct FixedBytes<let N: u8> {
    value: [Field; N],
}

// Bool, -> maybe no implementation needed
// Tuple(inner), -> maybe no implementation needed

fn extend_from_slice<let N: u8>(slice: [Field; N], out: &mut Vec<u8>) {
    for i in 0..N {
        out.push(slice[i] as u8);
    }
}

/// Encodes `token` as bytes into `out`.
fn encode_token<let N: u8>(token: Token<N>, out: &mut Vec<u8>, in_array: bool) {
    // Address(addr)
    if token.tag == 0 {
        if in_array {
            let slice = [0; 12];
            extend_from_slice(slice, out);
        }
        // extend_from_slice(token, addr.0);
    }
    // Int(n) | Uint(n)
    if token.tag == 1 {
        let n = token.integer.value;
        let buf = token.integer.value.to_be_bits(32);
        let start = if in_array {
            0
        } else {
            32 - ((n.to_be_bits(32).len() + 7) / 8) // WIP: unsure if `.bits()` is equal to `.to_be_bits(32).len()`
        };
        let mut buf_from_start: [Field; 32] = [0; 32];
        for i in start..32 {
            buf_from_start[i] = buf[i] as Field;
        }
        extend_from_slice(buf_from_start, out);
    }
    // Bool(b)
    if token.tag == 2 {
        if in_array {
            let slice = [0; 31];
            extend_from_slice(slice, out);
        }
        out.push(token.is_bool as u8);
    }
    // FixedBytes(bytes)
    if token.tag == 3 {
        let bytes = token.fixed_bytes.value;
        extend_from_slice(bytes, out);
        if in_array {
            let mut remaining: [Field; 32] = [0; 32];
            for i in 0..32 - bytes.len() {
                remaining[i] = 0;
            }
            extend_from_slice(remaining, out);
        }
    }

    // Encode dynamic types in-place, without their length
    // Bytes(bytes)
    if token.tag == 4 {}
    // String(s)
    if token.tag == 5 {}
    // Array(vec)
    if token.tag == 6 {}
}
