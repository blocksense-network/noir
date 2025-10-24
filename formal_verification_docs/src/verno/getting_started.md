# Getting Started with Verno

Welcome to Verno! This guide will walk you through the installation process and show you how to run Verno to formally verify your Noir programs.

## Installation

### Prerequisites

Before you begin, you need to have the Rust programming language and its package manager, Cargo, installed on your system.

We recommend using [rustup](https://rustup.rs/) to install and manage your Rust versions. This project includes a `rust-toolchain.toml` file, and `rustup` will automatically download and use the correct Rust toolchain for you.

### 1. Clone the Repository

First, clone the Verno repository from GitHub:

```bash
git clone https://github.com/blocksense-network/verno.git
cd verno
```

### 2. Set up the Environment

You have two options for setting up the development environment: using Nix or by running the provided shell scripts.

#### Option A: With Nix

If you have the [Nix package manager](https://nixos.org/download.html) installed, you can easily set up a development environment by running:

```bash
nix develop
```

This command will create a shell with all the necessary dependencies for Verno.

#### Option B: Without Nix

If you don't have Nix, you can set up the environment by running the following shell scripts from the root of the repository:

```bash
./get_verus_std.sh
./venir_build.sh
```

These scripts will download and build the required dependencies. Please follow any instructions printed by the scripts.

### 3. Build and Test

Once your environment is set up, you can build Verno using Cargo:

```bash
cargo build
```

To ensure everything is working correctly, run the test suite:

```bash
cargo test
```

If all tests pass, you have successfully installed Verno!

## Running Verno

Let's first create a Noir project in a new directory:

```bash
nargo new my_project
cd my_project
```

If you have `nargo` in your path or have set up the environment variable `NARGO_PATH` to contain the path to your `nargo` binary, you can actually run the following commands instead for creating a Noir project.


```bash
verno new my_project
cd my_project
```

Verno supports proxying commands to `nargo`.

Now, let's try running Verno on the following simple Noir program. Copy the following in `src/main.nr`:

```rust,ignore
#['requires(x >= 100)]
#['ensures(result >= 20)]
fn main(x: u32) -> pub u32 {
    let y = x / 5;
    y
}
```

To formally verify the code run the following command while inside of the project directory:

```bash
verno formal-verify
```

If the verification is successful, you should see the following output:

```
Verification successful!
```