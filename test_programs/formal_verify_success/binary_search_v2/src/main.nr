#[requires( (forall(|i, j| (0 <= i) & (i <= j) & (j < arr.len()) ==> arr[i] <= arr[j] )))]
#[ensures( (exists(|i| (0 <= i) & (i < arr.len()) & (arr[i] == find))) ==> result == true)]
#[ensures( (forall(|i| (0 <= i) & (i < arr.len()) & (arr[i] != find))) ==> result == false)]
fn main(arr: [u32; 200], find: u32) -> pub bool {
    let mut lind = 0;
    let mut hind = arr.len() - 1;
    let mut found = false;

    // Perform up to 5 iterations ((log2(200) ~= 7.64))
    for _ in 0..8 {
        let (new_lind, new_hind, new_found) = binary_search_step(arr, find, lind, hind, found);

        lind = new_lind;
        hind = new_hind;
        found = new_found;
    }

    found
}

#[requires(hind < arr.len())]
#[requires(lind <= arr.len())]
#[requires((lind > hind) ==> found == false)]
#[ensures(result.1 < arr.len())]
#[ensures(result.0 <= arr.len())]
#[ensures(found == true ==> ((result.0 == lind) & (result.1 == hind) & (result.2 == true) ))]
#[ensures(lind > hind ==> ((result.0 == lind) & (result.1 == hind) & (result.2 == false)) )]
#[ensures( (found == false) & (lind == hind) & (arr[(hind+lind) / 2] == find)  ==> (result.0 == lind) & (result.1 == hind) & (result.2 == true))]
#[ensures( (found == false) & (lind == hind) & (arr[(hind+lind) / 2] != find)  ==> (result.0 == lind) & (result.1 == hind) & (result.2 == false))]
#[ensures( ((found == false) & (lind < hind) & (arr[(hind+lind) / 2] == find)) 
                ==> (result.0 == lind) & (result.1 == hind) & (result.2 == true))]
#[ensures((found == false) & (lind < hind) & (arr[(hind + lind) / 2] < find) 
                      ==> ((result.0 == ((hind + lind) / 2) + 1) & (result.1 == hind) & (result.2 == false)) )]
#[ensures((found == false) & (lind < hind) & (arr[(hind + lind) / 2] > find) 
                      ==> ((result.0 == lind) & (result.1 == ((hind + lind) / 2)) & (result.2 == false)) )]
fn binary_search_step(arr: [u32; 200], find: u32, lind: u32, hind: u32, found: bool) -> (u32, u32, bool) {
    // found == true
    if found {
        (lind, hind, true)
    }
    else if lind >= hind {
        if (lind == hind) & (arr[(lind + hind) / 2] == find) {
            (lind, hind, true)
        } else {
            (lind, hind, found)
        }
    } else {
        let mid = (hind + lind) / 2;
        let num = arr[mid];

        if num == find {
            (lind, hind, true)
        } else if num < find {
            (mid + 1, hind, false)
        } else {
            (lind, mid, false)
        }
    }
}

