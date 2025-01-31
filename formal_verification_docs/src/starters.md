# Getting Started

## Installation
To get started with Noir FV you would have to use `git clone` to fetch the source code from our [Noir Github fork](https://github.com/blocksense-network/noir): 
```
git clone https://github.com/blocksense-network/noir.git
cd noir
git checkout formal-verification
```  
To build Noir FV using **Nix** as your package manager, execute the following commands:
```
direnv allow
cargo build
```
If you encounter any issues, refer to the [repository's README](https://github.com/blocksense-network/noir) for detailed instructions.
## Running Noir FV

Let's first create a Noir project in a new directory:
```
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
```
nargo fv
```
If the verification is successful, you should see the following output:
```
Verification successful!
```