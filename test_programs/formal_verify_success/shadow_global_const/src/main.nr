global x: u32 = 4;

// We are showing that 'x' here will be resolved as the parameter x of
// function `main` and not the global variable `x`.
#['ensures(result == x / 2)]
fn main(x: u32) -> pub u32 {
    x / 2 
}
