use dep::std;

global SHUFFLE_ROUND_COUNT = 90;
global FIRST_EIGHT_BYTES = 8;

fn field_from_bytes(bytes: [u8; FIRST_EIGHT_BYTES], big_endian: bool) -> Field {
    let mut as_field = 0;
    let mut offset = 1;
    for i in 0..FIRST_EIGHT_BYTES {
        let mut index = i;
        if big_endian {
            index = FIRST_EIGHT_BYTES - i - 1;
        }
        as_field += (bytes[index] as Field) * offset;
        offset *= 256;
    }

    as_field
}

fn compute_pivot(seed: [u8; 32], index_count: u64, current_round: u8) -> u64 {
    let mut round_and_seed_concatenated: [u8; 33] = [0;33];
    for i in 0..32 {
        round_and_seed_concatenated[i] = seed[i];
    }
    round_and_seed_concatenated[32] = current_round;

    let round_and_seed_hashed = std::hash::sha256(round_and_seed_concatenated);
    let mut round_and_seed_hashed_first_eight_bytes: [u8; 8] = [0;8];
    for i in 0..8 {
        round_and_seed_hashed_first_eight_bytes[i] = round_and_seed_hashed[i];
    }
    let round_and_seed_hashed_first_eight_bytes = field_from_bytes(round_and_seed_hashed_first_eight_bytes, false) as u64;
    round_and_seed_hashed_first_eight_bytes % index_count
}

fn compute_flip(pivot: u64, index_count: u64, index: u64) -> u64 {
    (pivot + index_count - index) % index_count
}

fn compute_position(index: u64, flip: u64) -> u64 {
    if index <= flip {
        flip
    } else {
        index
    }
}

fn compute_source(seed: [u8; 32], current_round: u8, position: u64) -> [u8; 32] {
    let position = position / 256;
    let mut source: [u8; 41] = [0; 41];
    for i in 0..32 {
        source[i] = seed[i];
    }
    source[32] = current_round;
    for i in 0..8 {
        source[33 + i] = position as u8;
    }
    let source_hashed = std::hash::sha256(source);
    source_hashed
}

fn compute_byte(source: [u8; 32], position: u64) -> u8 {
    source[(position % 256) / 8]
}

fn compute_bit(byte: u8, position: u64) -> u8 {
    (byte >> (position % 8) as u8) % 2
}

fn compute_index(index: u64, bit: u8, flip: u64) -> u64 {
    if bit == 1{
        flip
    } else {
        index
    }
}

// reference: https://github.com/LeastAuthority/eth2.0-specs/blob/dev/specs/phase0/beacon-chain.md#compute_shuffled_index
fn compute_shuffled_index(index: u64, index_count: u64, seed: [u8; 32]) -> u64 {
    let mut index = index;
    assert(index as u8 < index_count as u8);

    for current_round in 0..SHUFFLE_ROUND_COUNT {
        let pivot = compute_pivot(seed, index_count, current_round);
        let flip = compute_flip(pivot, index_count, index);
        let position = compute_position(index, flip);
        let source = compute_source(seed, current_round, position);
        let byte = compute_byte(source, position);
        let bit = compute_bit(byte, position);
        index = compute_index(index, bit, flip);
    }
    index
}