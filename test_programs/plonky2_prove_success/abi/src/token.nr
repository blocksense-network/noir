global ADDRESS_BYTES: Field = 20;
global MAX_BYTES: u8 = 32;
type _Array = Field;

struct Wrapper {
    is_array: bool,
    /// Ethereum ABI params
    token: Token,
    /// Array of params with unknown size.
    ///
    /// solidity name eg. int[], bool[], address[5][]
    array: Array,
    /// Array with known size.
    ///
    /// solidity name eg.: int[3], bool[3], address[][8]
    /// Encoding of array is equal to encoding of consecutive elements of array.
    fixed_array: FixedArray,
}

struct Token {
    tag: Field,
    /// Address.
    ///
    /// solidity name: address
    /// Encoded to left padded [0u8; 32].
    address: Address,
    /// Vector of bytes with known size.
    ///
    /// solidity name eg.: bytes8, bytes32, bytes64, bytes1024
    /// Encoded to right padded [0u8; ((N + 31) / 32) * 32].
    fixed_bytes: FixedBytes,
    /// Vector of bytes of unknown size.
    ///
    /// solidity name: bytes
    /// Encoded in two parts.
    /// Init part: offset of 'closing part`.
    /// Closing part: encoded length followed by encoded right padded bytes.
    bytes: Bytes,
    /// Unsigned integer.
    ///
    /// solidity name: uint
    integer: Integer,
    /// Boolean value.
    ///
    /// solidity name: bool
    /// Encoded as left padded [0u8; 32], where last bit represents boolean value.
    is_bool: Bool,
    /// String.
    ///
    /// solidity name: string
    /// Encoded in the same way as bytes. Must be utf8 compliant.
    string: String,
}

struct Address {
    value: str<ADDRESS_BYTES>,
}

struct Integer {
    value: Field, // *TODO: Replace Field with Bigint.
}

struct String {
    value: Vec<u8>,
}

struct Array {
    value: Vec<Token>,
}

struct FixedArray {
    value: Vec<Token>,
}

struct Bytes {
    value: Vec<u8>,
}

struct Bool {
    value: bool,
}

struct FixedBytes {
    value: Vec<u8>,
}

struct EncodedBytes {
    value: Vec<u8>,
}

impl EncodedBytes {
    fn extend_from_slice(self: &mut Self, slice: [u8]) {
        for i in 0..slice.len() {
            self.value.push(slice[i] as u8);
        }
    }

    fn extend_from_vec(self: &mut Self, vec: Vec<u8>) {
        for i in 0..vec.len() {
            self.value.push(vec.get(i));
        }
    }
}

fn encode_packed(tokens: [Token]) -> Vec<u8> {
    // Get vec capacity and find invalid tokens
    let mut max = 0;
    for i in 0..tokens.len() {
        let token = tokens[i];
        check(token);
        max += max_encoded_length(token);
    }
    // Encode the tokens
    let mut bytes = Vec::new();
    for i in 0..tokens.len() {
        let token = tokens[i];
        // encode_token(token, &mut bytes, false);
    }

    bytes
}

/// The maximum byte length of the token encoded using packed mode.
fn max_encoded_length(token: Token) -> u8 {
    if token.tag == 0 {
        20
    } else if (token.tag == 1) | (token.tag == 3) {
        32
    } else if token.tag == 2 {
        1
    } else if token.tag == 4 {
        token.bytes.value.len() as u8
    } else if token.tag == 5 {
        token.string.value.len() as u8
    } else {
        // let mut sum = 0;
        // for i in 0..token.array.value.len() {
        //     let token = token.array.value.get(i);
        //     sum += max_encoded_length(token);
        // }
        // sum
        0
    }
}

/// Tuples and nested arrays are invalid in packed encoding.
fn check(token: Token) {
    if token.tag == 4 {
        assert(
            token.bytes.value.len() as u8 > MAX_BYTES, "Storage slot 0 not allowed. Storage slots must start from 1."
        );
    }
    else if token.tag == 123123123123 {}
    // Handle Tuple!
    else {
        // for i in 0..token.array.value.len() {
        //     let token = token.array.value.get(i);
        //     check(token);
        // }
    }
}

// Benchmark if it's actually faster than the `extend_from_slice` function
// WIP: Extend from a dynamic slice (up to a max size) into an output buffer
fn extend_from_slice_<let N: u8>(slice: [Field], out: &mut [u8; 32], out_len: &mut Field) {
    for i in 0..slice.len() {
        if (*out_len) as u8 < 32 {
            out[*out_len] = slice[i] as u8;
            *out_len += 1;
        }
    }
}

/// Encodes `token` as bytes into `out`.
fn encode_token(wrapper: Wrapper, out: &mut EncodedBytes, in_array: bool) {
    if !wrapper.is_array {
        // Padded to 32 bytes if in_array
        // if else is faster
        // Address
        if wrapper.token.tag == 0 {
            if in_array {
                let buf: [u8] = &[0; MAX_BYTES - ADDRESS_BYTES];
                out.extend_from_vec(Vec::from_slice(buf));
            }
            out.extend_from_vec(wrapper.token.address.value.as_bytes_vec());
        }
        // Int | Uint
        if wrapper.token.tag == 1 {
            let buf = wrapper.token.integer.value.to_be_bytes(32);

            let start = if in_array {
                0
            } else {
                32 - ((wrapper.token.integer.value.to_be_bits(256).len() + 7) / 8)
            };
            for i in start..32 {
                out.value.push(buf[i]);
            }
        }
        // Bool
        if wrapper.token.tag == 2 {
            if in_array {
                let buf: [u8] = [0; MAX_BYTES - 1];
                out.extend_from_vec(Vec::from_slice(buf));
            }
            out.value.push(wrapper.token.is_bool.value as u8);
        }
        // FixedBytes
        if wrapper.token.tag == 3 {
            out.extend_from_vec(wrapper.token.fixed_bytes.value);
            if in_array {
                for _ in 0..MAX_BYTES - wrapper.token.fixed_bytes.value.len() as u8 {
                    out.value.push(0);
                }
            }
        }
        // Encode dynamic types in-place, without their length
        // Bytes
        if wrapper.token.tag == 4 {
            out.extend_from_vec(wrapper.token.bytes.value);
        }
        // String
        if wrapper.token.tag == 5 {
            out.extend_from_vec(wrapper.token.string.value);
        }
    } else {
        // Array | FixedArray
        for i in 0..wrapper.array.value.len() {
            let token = wrapper.array.value.get(i);
            let vector: Vec<Token> = Vec::new(); // WIP
            encode_token(
                Wrapper { is_array: true, token, array: Array { value: vector }, fixed_array: FixedArray { value: vector } },
                out,
                true
            );
        }
    }
}
