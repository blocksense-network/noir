#[requires( (forall(|i, j| (0 <= i) & (i <= j) & (j < arr.len()) ==> arr[i] <= arr[j] )))]
#[ensures( (exists(|i| (0 <= i) & (i < arr.len()) & (arr[i] == find))) ==> result == true)]
#[ensures( (forall(|i| (0 <= i) & (i < arr.len()) & (arr[i] != find))) ==> result == false)]
fn main(arr: [u32; 30], find: u32) -> pub bool {
    let mut lind = 0;
    let mut hind = arr.len() - 1;
    let mut found = false;

    // Perform up to 6 iterations ((log2(30) ~= 5) + 1)
    for _ in 0..6 {
        // assert((lind > hind) ==> found == false);

        let (new_lind, new_hind, new_found) = binary_search_step(arr, find, lind, hind, found);
        // assert((arr[(lind + hind) / 2] == find) ==> new_found == true);

        if (check(new_lind, new_hind)) {
            lind = new_lind;
            hind = new_hind;
            found = new_found;
        } else {
            found = false;
        }
    }

    found
}

#[ensures(lind <= hind ==> result == true)]
#[ensures(lind > hind ==> result == false)]
fn check(lind: u32, hind: u32) -> bool {
    if (lind <= hind) { true } else { false }
}

#[requires(hind < arr.len())]
#[requires(lind <= arr.len())]
#[requires((lind > hind) ==> found == false)]
#[ensures((found == true) ==> ((result.0 == lind) & (result.1 == hind) & (result.2 == true) ) )]
#[ensures((lind > hind) ==> ((result.0 == lind) & (result.1 == hind) & (result.2 == false)) )]
#[ensures((lind <= hind) & (arr[(hind + lind) / 2] == find) 
                      ==> ((result.0 == lind) & (result.1 == hind) & (result.2 == true)) )]
#[ensures((found == false) & (lind <= hind) & (((hind + lind) / 2) != 0) & (arr[(hind + lind) / 2] < find) 
                      ==> ((result.0 == ((hind + lind) / 2) + 1) & (result.1 == hind) & (result.2 == false)) )]
#[ensures((found == false) & (lind <= hind) & (arr[(hind + lind) / 2] > find) & (((hind + lind) / 2) != 0) 
                     ==> ((result.0 == lind) & (result.1 == (((hind + lind) / 2) - 1)) & (result.2 == false)) )]
#[ensures((found == false) & (lind <= hind) & (arr[(hind + lind) / 2] > find) & (((hind + lind) / 2) == 0) 
                     ==> ((result.0 == lind) & (result.1 == hind) & (result.2 == false)) )]
#[ensures((found == false) & (lind <= hind) & (arr[(hind + lind) / 2] < find) & (((hind + lind) / 2) == 0) 
                     ==> ((result.0 == lind + 1) & (result.1 == hind) & (result.2 == false)) )]
#[ensures(result.1 < arr.len())]
#[ensures(result.0 <= arr.len())]
fn binary_search_step(arr: [u32; 30], find: u32, lind: u32, hind: u32, found: bool) -> (u32, u32, bool) {
    // found == true
    if found & lind <= hind {
        (lind, hind, true)
    } else if lind <= hind {
        let mid = (hind + lind) / 2;
        let num = arr[mid];

        if num == find {
            (lind, hind, true)
        } else if num < find {
            (mid + 1, hind, false)
        } else if mid == 0 {
            (lind, hind, false)
        } else {
            (lind, mid - 1, false)
        }
    } else {
        (lind, hind, false)
    }
}
