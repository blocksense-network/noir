# Getting Started

## Installation
<div class="warning">  
NOTE: This installation requires more than 20GBs of drive memory.  
</div>  

First, fetch the source code from our [Noir Github fork](https://github.com/blocksense-network/noir):

```bash
git clone https://github.com/blocksense-network/noir.git -b formal-verification
cd noir
```

Nix-powered machines with `direnv` only need to do:

```bash
direnv allow
cargo build
```

Nix-powered machines without `direnv` have to do the following:

```bash
nix develop
cargo build
export PATH=$PATH:$PWD/target/debug
```

For other systems, you will need to have `rustup` installed. Follow those instructions:

```bash
git clone git@github.com:blocksense-network/Venir.git

pushd Venir
export RUSTC_BOOTSTRAP=1;
cargo build
export LD_LIBRARY_PATH="${HOME}/.rustup/toolchains/1.76.0-x86_64-unknown-linux-gnu/lib:${LD_LIBRARY_PATH}"
export PATH=$PATH:$PWD/target/debug

popd
mkdir lib

git clone git@github.com:Aristotelis2002/verus-lib.git
pushd verus-lib
cargo build
cp ./target/debug/vstd.vir ../lib/

popd
cargo build
export PATH=$PATH:$PWD/target/debug
```

If you encounter any issues, refer to the [repository's README](https://github.com/blocksense-network/noir) for detailed instructions.

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
