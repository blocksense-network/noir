global ADDRESS_BYTES: Field = 42;
global MAX_BYTES: u8 = 32;

type TokenTuple = Vec<Field>;

struct Token {
    kind: TokenKind,
}

struct TokenKind {
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
    /// Array of params with unknown size.
    ///
    /// solidity name eg. int[], bool[], address[5][]
    array: Array,
    /// Array with known size.
    ///
    /// solidity name eg.: int[3], bool[3], address[][8]
    /// Encoding of array is equal to encoding of consecutive elements of array.
    fixed_array: FixedArray,
    /// Tuple of params of variable types.
    ///
    /// solidity name: tuple
    tuple: TokenTuple,
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
    value: Vec<Field>,
}

struct FixedArray {
    value: Vec<Field>,
}

struct Tuple {
    value: Vec<Field>,
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
    fn new() -> Self {
        Self { value: Vec::new() }
    }

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
    let mut bytes = EncodedBytes::new();
    encode_token(tokens, 0, &mut bytes, false);

    bytes.value
}

/// The maximum byte length of the token encoded using packed mode.
fn max_encoded_length(token: Token) -> u8 {
    if token.kind.tag == 0 {
        20
    } else if (token.kind.tag == 1) | (token.kind.tag == 3) {
        32
    } else if token.kind.tag == 2 {
        1
    } else if token.kind.tag == 4 {
        token.kind.bytes.value.len() as u8
    } else if token.kind.tag == 5 {
        token.kind.string.value.len() as u8
    } else {
        // let mut sum = 0;
        // for i in 0..token.kind.array.value.len() {
        //     let token = token.kind.array.value.get(i);
        //     if
        //     sum += max_encoded_length(token);
        // }
        // sum
        0
    }
}

/// Tuples and nested arrays are invalid in packed encoding.
fn check(token: Token) {
    if token.kind.tag == 4 {
        assert(token.kind.bytes.value.len() as u8 > MAX_BYTES, "Invalid bytes length");
    } else if token.kind.tag == 7 {
        assert(token.kind.tag != 7, "Invalid token");
    } else {
        for i in 0..token.kind.array.value.len() {
            check(token);
        }
    }
    // }
}

fn is_dynamic(token: Token) -> bool {
    let token_tag = token.kind.tag;
    if (token_tag == 4) | (token_tag == 5) | (token_tag == 6) {
        true
    } else if (token_tag) == 7 {
        is_dynamic(token) // FixedArray
    } else if (token_tag) == 8 {
        is_dynamic(token) // Tuples
    } else {
        false
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

fn encode_token(tokens: [Token], index: u64, out: &mut EncodedBytes, is_in_array: bool) {
    let token = tokens[index];
    // Padded to 32 bytes if in_array
    // if else is faster
    // Address
    if token.kind.tag == 0 {
        if is_in_array {
            let buf: [u8] = &[0; MAX_BYTES - ADDRESS_BYTES];
            out.extend_from_vec(Vec::from_slice(buf));
        }
        out.extend_from_vec(token.kind.address.value.as_bytes_vec());
    }
    // Int | Uint
    else if token.kind.tag == 1 {
        let buf = token.kind.integer.value.to_be_bytes(32);
        let start = if is_in_array {
            0
        } else {
            32 - ((token.kind.integer.value.to_be_bits(256).len() + 7) / 8)
        };
        for i in start..32 {
            out.value.push(buf[i]);
        }
    }
    // Bool
    else if token.kind.tag == 2 {
        if is_in_array {
            let buf: [u8] = [0; MAX_BYTES - 1];
            out.extend_from_vec(Vec::from_slice(buf));
        }
        out.value.push(token.kind.is_bool.value as u8);
    }
    // FixedBytes
    else if token.kind.tag == 3 {
        out.extend_from_vec(token.kind.fixed_bytes.value);
        if is_in_array {
            for _ in 0..MAX_BYTES - token.kind.fixed_bytes.value.len() as u8 {
                out.value.push(0);
            }
        }
    }
    // Encode dynamic types in-place, without their length
    // Bytes
    else if token.kind.tag == 4 {
        out.extend_from_vec(token.kind.bytes.value);
    }
    // String
    else if token.kind.tag == 5 {
        out.extend_from_vec(token.kind.string.value);
    } else if token.kind.tag == 6 {
        // Array | FixedArray
        for elem_index in 0..token.kind.array.value.len() {
            encode_token(tokens, elem_index as u64, out, true);
        }
    }
}

fn create_new_token() {
    let token_a = "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48";
    let kind = TokenKind {
        tag: 0,
        tuple: Vec::new(),
        address: Address { value: token_a },
        fixed_bytes: FixedBytes { value: Vec::new() },
        bytes: Bytes { value: Vec::new() },
        integer: Integer { value: 0 },
        is_bool: Bool { value: false },
        string: String { value: Vec::new() },
        array: Array { value: Vec::new() },
        fixed_array: FixedArray { value: Vec::new() }
    };
    let token = Token { kind };
}
