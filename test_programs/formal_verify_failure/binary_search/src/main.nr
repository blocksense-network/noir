fn main(arr: [u32; 100], find: u32) -> pub bool {
    let mut lind = 0;
    let mut hind = arr.len() - 1;
    let mut found = false;

    // Perform up to 7 iterations (log2(100) ~= 7)
    for _ in 0..7 {
        let (new_lind, new_hind, new_found) = binary_search_step(arr, find, lind, hind, found);

        lind = new_lind;
        hind = new_hind;
        found = new_found;
    }

    found
}

#[requires(hind < arr.len())]
#[requires((lind <= hind) & (hind > 1) )]
fn binary_search_step(
    arr: [u32; 100],
    find: u32,
    lind: u32,
    hind: u32,
    found: bool
) -> (u32, u32, bool) {
    // Only proceed if the value hasn't been found and the search range is valid
    if !found & lind <= hind {
        let half = (hind + lind) / 2;
        let current = arr[half];
        if current == find {
            (lind, hind, true)
        } else if current < find {
            (half + 1, hind, false)
        } else {
            (lind, half - 1, false)
        }
    } else {
        (lind, hind, found)
    }
}
