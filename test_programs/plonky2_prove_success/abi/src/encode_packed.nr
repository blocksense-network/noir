// Placeholder function to encode static values
func encodeStatic(value: [u8], writer: [u8]) -> [u8] {
    // Encode static value into writer
    // (Placeholder implementation)
    writer.append(value)
}

// Placeholder function to encode dynamic values
func encodeDynamic(value: [u8], writer: [u8]) -> (u32, [u8]) {
    // Encode dynamic value into writer
    // (Placeholder implementation)
    let offset = writer.len()
    let new_writer = writer.append(value)
    return (offset, new_writer)
}

struct Writer {
}

struct Coder {
}

// Main pack function
fn pack(writer: Writer, coders: Coder, values: [u8]) -> u32 {
    let mut arrayValues: [u8] = []
    
    // Handle values depending on type (Array or Object)
    if values.is_array() {
        arrayValues = values
    } else if values.is_object() {
        let mut unique: [u8] = []
        arrayValues = coders.map(|coder| {
            let name = coder.localName()
            assert(name != None, "Missing name in coder")
            assert(!unique.contains(name), "Duplicate name in coder")
            unique.append(name)
            values[name]
        })
    } else {
        assert(false, "Invalid tuple value")
    }
    
    assert(coders.len() == arrayValues.len(), "Types/value length mismatch")

    let mut staticWriter: [u8] = []
    let mut dynamicWriter: [u8] = []
    
    // Encode each coder
    for (index, coder) in coders.iter().enumerate() {
        let value = arrayValues[index]
        
        if coder.is_dynamic() {
            let (dynamicOffset, updatedWriter) = encodeDynamic(value, dynamicWriter)
            dynamicWriter = updatedWriter
            let updateFunc = |baseOffset: u32| {
                // Placeholder for updating offsets
            }
            // Add update function for dynamic offset
        } else {
            staticWriter = encodeStatic(value, staticWriter)
        }
    }
    
    // Backfill all dynamic offsets
    // Placeholder logic to apply updates to offsets
    
    let totalLength = writer.len() + staticWriter.len() + dynamicWriter.len()
    writer.append(staticWriter)
    writer.append(dynamicWriter)
    return totalLength
}


/// Encodes `token` as bytes into `out`.
fn encode_token(token: Token, out: &mut EncodedBytes, in_array: bool) {
    if !token.kind.is_array {
        // Padded to 32 bytes if in_array
        // if else is faster
        // Address
        if token.kind.tag == 0 {
            if in_array {
                let buf: [u8] = &[0; MAX_BYTES - ADDRESS_BYTES];
                out.extend_from_vec(Vec::from_slice(buf));
            }
            out.extend_from_vec(token.kind.address.value.as_bytes_vec());
        }
        // Int | Uint
        else if token.kind.tag == 1 {
            let buf = token.kind.integer.value.to_be_bytes(32);
            let start = if in_array {
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
            if in_array {
                let buf: [u8] = [0; MAX_BYTES - 1];
                out.extend_from_vec(Vec::from_slice(buf));
            }
            out.value.push(token.kind.is_bool.value as u8);
        }
        // FixedBytes
        else if token.kind.tag == 3 {
            out.extend_from_vec(token.kind.fixed_bytes.value);
            if in_array {
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
        }
    } else {
        // Array | FixedArray
        for i in 0..token.kind.array.value.len() {
            encode_token(token, out, true);
        }
    }
}

trait Coder {}
struct ArrayCoder { /* fields */ }
struct TupleCoder { /* fields */ }
struct AddressCoder { /* fields */ }
struct BooleanCoder { /* fields */ }
struct StringCoder { /* fields */ }
struct BytesCoder { /* fields */ }
struct NullCoder { /* fields */ }
struct NumberCoder { /* fields */ }
struct FixedBytesCoder { /* fields */ }

// -> convert type e.g. str<N> to bytes and then insert it as a parameter
// -> values are in bytes
fn encode(types: Vec<u8>, values: Vec<u8>) {
    assert(types.len() == values.len(), "types/values length mismatch");

    for i in 0..types.len() {}
    let coders = 0;
}
