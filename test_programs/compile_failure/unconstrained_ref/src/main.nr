unconstrained fn uncon_ref() -> &mut Field {
    let lr = &mut 7;
    lr
}

fn main() {
    let e = unsafe {
        uncon_ref()
    };
}
