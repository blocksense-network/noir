global ADDRESS_BYTES: Field = 42;
global MAX_BYTES: u8 = 32;

struct Token {
    kind: Field,
    primitive: PrimitiveType,
    array: Vec<PrimitiveType>,
    tuple: Vec<PrimitiveType>,
}

fn check(token: Token) {
    if token.kind == 6 {
        assert(token.primitive.value.len() > 32, "Invalid bytes length");
    } else if token.kind == 7 {
        assert(false, "Invalid token");
    } else if token.kind == 5 {
        if token.array.get(0).is_dynamic() {
            assert(false);
        }
    }
}

impl PrimitiveType {
    fn is_dynamic(token: Self) -> bool {
        if (token.tag == 0) | (token.tag == 1) | (token.tag == 3) {
            false
        } else {
            assert(false);
            false
        }
    }

    fn max_encoded_length(self: Self) -> u32 {
        if self.tag == 0 {
            20
        } else if (self.tag == 1) | (self.tag == 6) {
            32
        } else if self.tag == 3 {
            1
        } else if self.tag == 4 {
            self.bytes.value.len()
        } else if self.tag == 2 {
            self.string.value.len()
        } else {
            0
        }
    }
}

/// The maximum byte length of the token encoded using packed mode.
fn max_encoded_length(token: Token) -> u8 {
    if token.kind == 0 {
        20
    } else if (token.kind == 1) | (token.kind == 6) {
        32
    } else if token.kind == 3 {
        1
    } else if token.kind == 4 {
        token.primitive.bytes.value.len() as u8
    } else if token.kind == 2 {
        token.primitive.string.value.len() as u8
    } else {
        let mut sum = 0;
        for i in 0..token.array.len() {
            let current_primitive = token.array.get(i);
            sum += current_primitive.max_encoded_length();
        }
        sum as u8
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

// 0
struct Address {
    value: str<ADDRESS_BYTES>,
    is_valid: bool,
}

// 1
struct Integer {
    value: Field, // *TODO: Replace Field with Bigint.
    is_valid: bool,
}

// 2
struct String {
    value: Vec<u8>,
    is_valid: bool,
}

// 3
struct Bool {
    value: bool,
    is_valid: bool,
}

// 4
struct Bytes {
    value: Vec<u8>,
    is_valid: bool,
}

// 5
struct Array {
    value: Vec<Field>,
    is_valid: bool,
}

// 6
struct FixedBytes {
    value: Vec<u8>,
    is_valid: bool,
}

// 7
struct Tuple {
    value: Vec<Field>,
    is_valid: bool,
}

struct PrimitiveType {
    tag: u8,
    value: Vec<u8>,
    address: Address,
    integer: Integer,
    is_bool: Bool,
    fixed_bytes: FixedBytes,
    bytes: Bytes,
    string: String,
    array: Array,
    tuple: Tuple,
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

fn encode_token(tokens: [Token], index: u64, out: &mut EncodedBytes, is_in_array: bool) {
    let token = tokens[index];
    // Padded to 32 bytes if in_array
    // if else is faster
    // Address
    if token.kind == 0 {
        if is_in_array {
            let buf: [u8] = &[0; MAX_BYTES - ADDRESS_BYTES];
            out.extend_from_vec(Vec::from_slice(buf));
        }
        out.extend_from_vec(token.primitive.address.value.as_bytes_vec());
    }
    // Int | Uint
    else if token.kind == 1 {
        let buf = token.primitive.integer.value.to_be_bytes(32);
        let start = if is_in_array {
            0
        } else {
            32 - ((token.primitive.integer.value.to_be_bits(256).len() + 7) / 8)
        };
        for i in start..32 {
            out.value.push(buf[i]);
        }
    }
    // Bool
    else if token.kind == 3 {
        if is_in_array {
            let buf: [u8] = [0; MAX_BYTES - 1];
            out.extend_from_vec(Vec::from_slice(buf));
        }
        out.value.push(token.primitive.is_bool.value as u8);
    }
    // FixedBytes
    else if token.kind == 6 {
        out.extend_from_vec(token.primitive.fixed_bytes.value);
        if is_in_array {
            for _ in 0..MAX_BYTES - token.primitive.fixed_bytes.value.len() as u8 {
                out.value.push(0);
            }
        }
    }
    // Encode dynamic types in-place, without their length
    // Bytes
    else if token.kind == 4 {
        out.extend_from_vec(token.primitive.bytes.value);
    }
    // String
    else if token.kind == 2 {
        out.extend_from_vec(token.primitive.string.value);
    } else if token.kind == 5 {
        // Array
        for elem_index in 0..token.primitive.array.value.len() {
            encode_token(tokens, elem_index as u64, out, true);
        }
    }
}
