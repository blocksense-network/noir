# Blocksense Formal Verification in Noir

We want to provide smart contract engineers with powerful tools that they can use to write safe, secure and reliable code. This is the reason why blocksense is developing the Noir PLONKY2 back-end. However, another important system for ensuring program results are formal verifications.

## What is formal verification?

With unit tests, software can be verified according to a number of specific and hand-mande inputs. On the other hand, formal verification is able to mathematically ensure that all possible scenarios are accounted for, thus covering more possibilities than hand-made test normally would and eliminating entire categories of bugs.

## Why we use Verus as a back-end

We chose Verus as the back-end for implementing formal verification in Noir due to its architecture being well-suited for our needs, reducing the complexity of incorporating it into Noir while supporting nearly all the features required for our prototype. Influenced by tools like Dafny and AdaSpark, Verus integrates the Z3 SMT solver, enabling precise reasoning and verification of logical constraints.

## How to install

0. Install dependencies, you'll only need two things: the [nix package manager](https://nixos.org/download/) and [direnv](https://direnv.net/docs/installation.html). They're compatible with most OSes and will **not** collide with your system.

> [!IMPORTANT]
> After installing `direnv` do not forget to [add the hook](https://direnv.net/docs/hook.html)!

1. Clone [our branch](https://github.com/blocksense-network/noir/tree/formal-verification) with SSH:

    ```bash
    git clone git@github.com:blocksense-network/noir.git -b formal-verification
    ```

2. Navigate to the folder `noir`.

    ```bash
    cd noir
    ```

3. Run direnv command:

    ```bash
    direnv allow
    ```

    This should result in a lot of things happening. If not, you haven't [added the direnv hook](https://direnv.net/docs/hook.html)!

> [!WARNING]
> Depending on your `nix` installation, you may get a `Permission denied` error. In that case, it's best to start a superuser shell and continue from there:
> 
> ```bash
> sudo su                      # Start superuser shell
> eval "$(direnv hook bash)"   # Setup the direnv hook
> direnv allow
> ```

4. Test if everything works:

    ```bash
    cargo test formal
    ```

    This will also take a little bit of time, until the project fully compiles.

## Example usage

> [!CAUTION]
> The Noir formal-verifications project is a prototype! Expect to find bugs and limitations!

1. Create a new project:

    ```bash
    nargo new my_program
    ```

2. Navigate to the folder:

    ```bash
    cd my_program
    ```

3. Update `src/main.nr` with your favorite text editor to:

    ```noir
    #[requires(x < 100 & 0 < y & y < 100)]
    #[ensures(result >= 5 + x)]
    fn main(x: u32, y: u32) -> pub u32 {
        x + y * 5
    }
    ```

4. Finally, verify the program:

    ```bash
    nargo formal-verify
    ```

## Leveraging the formal verification

We examine the following code snippet:
```noir
fn main(x: i32, y:i32, arr: [u32; 5]) -> pub u32 {
  let z = arithmetic_magic(x, y);
  arr[z]
}

fn arithmetic_magic(x: i32, y: i32) -> i32 {
    (x / 2) + (y / 2)
}
```
Formally verifying it produces an error.

This is due to us not ensuring that `z` stays in bounds of the `arr` array.

Adding an if statement which checks for the aforementioned scenario resolves the error.  
The following formally verifies successfully:
```noir
fn main(x: i32, y:i32, arr: [u32; 5]) -> pub u32 {
  let z = arithmetic_magic(x, y);
  if (z >= 0) & (z < 5) {
    arr[z]
  } else {
      0
    }
}

fn arithmetic_magic(x: i32, y: i32) -> i32 {
    (x / 2) + (y / 2)
}
```
