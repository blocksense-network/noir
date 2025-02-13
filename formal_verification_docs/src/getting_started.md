# Getting Started

## Installation
<div class="warning">  
NOTE: This installation requires more than 20GBs of drive memory.  
</div>  

First, fetch the source code from our [Noir Github fork](https://github.com/blocksense-network/noir):

```bash
git clone https://github.com/blocksense-network/noir.git -b formal-verification
```

* Nix-powered machines with `direnv` only need to do:

    ```bash
    cd noir
    direnv allow
    cargo build
    ```

* Nix-powered machines without `direnv` have to do the following:

    ```bash
    cd noir
    nix develop
    cargo build
    export PATH=$PATH:$PWD/target/debug
    ```

* For other systems, you will need to have `rustup` installed. Follow those instructions:

    ```bash
    git clone https://github.com/blocksense-network/Venir.git

    pushd Venir
    export RUSTC_BOOTSTRAP=1
    cargo build
    export LD_LIBRARY_PATH="${HOME}/.rustup/toolchains/1.76.0-x86_64-unknown-linux-gnu/lib:${LD_LIBRARY_PATH}"
    export PATH=$PATH:$PWD/target/debug
    popd

    pushd noir
    mkdir lib
    popd

    git clone https://github.com/Aristotelis2002/verus-lib.git
    
    pushd verus-lib/source/
    pushd tools
    ./get-z3.sh
    export VERUS_Z3_PATH=$PWD/z3
    popd
    export RUSTC_BOOTSTRAP=1
    export VERUS_IN_VARGO=1
    export RUSTFLAGS="--cfg proc_macro_span --cfg verus_keep_ghost --cfg span_locations"
    cargo build
    cargo run -p vstd_build -- ./target/debug/
    cp ./target/debug/vstd.vir ../../noir/lib/
    popd

    pushd noir
    cargo build
    export PATH=$PATH:$PWD/target/debug
    ```

If you encounter any issues, refer to our [Github discussions](https://github.com/blocksense-network/noir/issues). If it's not listed, don't hesitate to report it—we're happy to assist

## Running Noir FV

Let's first create a Noir project in a new directory:

```bash
nargo new my_project
cd my_project
```

Now, let's try running Noir FV on the following simple Noir program:

```rust,ignore
#[requires(x >= 100)]
#[ensures(result >= 20)]
fn main(x: u32) -> pub u32 {
    let y = x / 5;
    y
}
```

To formally verify the code run the following command while inside of the project directory:

```bash
nargo fv
```

If the verification is successful, you should see the following output:

```
Verification successful!
```
