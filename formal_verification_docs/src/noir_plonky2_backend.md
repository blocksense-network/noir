# Noir PLONKY2 Backend

An essential component of Blocksense are ZK proofs, the primary technology which eliminates bad actors from manipulating truth. In order to make it easier for our ZK engineers to develop this component, we built a PLONKY2 backend for the Noir programming language. While this backend is not completely stable, it already serves as a good proof-of-concept. Since our work is public and open-source, anyone can download it, try it out, and submit feedback.

## What is zero-knowledge?

ZK is a method by which any program can have its execution and results mathematically and publicly verified, without exposing any private data. Thus, *anyone* can be sure some sort of execution was done correctly and honestly, even if any number of secrets were used as input.

The difficulty with this system comes from having to transform arbitrary programs into mathematical expressions (circuits). Since the vast majority of languages are not created with ZK in mind, it's almost impossible to do this step.

### What is Noir?

Noir is a programming language, designed by [Aztec Labs](https://aztec.network/) from the ground up for ZK. Its syntax is based on a small subset of Rust, with all of the limitations which make conversion to circuits possible. To be more precise, code is compiled down to ACIR circuits, which can then be converted to any proving system's native circuits.

The only system which has been adapted for ACIR is barratenberg, also built by Aztec Labs. While it is an impressive project, we wanted to experiment with different proving systems in order to leverage the latest and greatest of ZK research. This is why we built our PLONKY2 backend for the Noir programming language.

### What is PLONKY2?

PLONKY2 is a zkSNARK built by [Polygon Labs](https://polygon.technology/), with efficiency, decomposition and size in mind. Recursive proofs can be generated faster than other systems. This enables proofs to be split into subproofs and distributed across hundreds or thousands of machines, and it provides the ability to shrink proofs down dramatically in seconds.

A simple programming language to write ZK programs, with fast-to-generate, distributed and small in size proofs gives us the best of both worlds. The consensus mechanism can be developed and maintained without much difficulty, while it's execution can be distributed on the blockchain with vast assuredness of the result's correctness, all for a small cost.

## Installing

0. Install dependencies, you'll only need two things: the [nix package manager](https://nixos.org/download/) and [direnv](https://direnv.net/docs/installation.html). They're compatible with most OSes and will **not** collide with your system.

    After installing `direnv` do not forget to [add the hook](https://direnv.net/docs/hook.html)!

1. Clone [our repository](https://github.com/blocksense-network/noir/) with SSH:

    ```bash copy
    git clone https://github.com/blocksense-network/noir.git
    ```

2. Navigate to the folder `noir`.

    ```bash copy
    cd noir
    ```

3. Run direnv command:

    ```bash copy
    direnv allow
    ```

    Depending on your `nix` installation, you may get a `Permission denied` error. In that case, it's best to start a superuser shell and continue from there:

    ```bash
    sudo su                      # Start superuser shell
    eval "$(direnv hook bash)"   # Setup the direnv hook
    direnv allow
    ```

    This should result in a plethora of things happening in the background and foreground. Sit back, relax, and wait it out. By the end you'll have everything ready to start work.

4. Test if everything works:

    ```bash copy
    cargo test zk_dungeon
    ```

    This will also take a little bit of time, until the project fully compiles.

## Using

We're now ready to create our first proof!

1. Create a new project:

    ```bash copy
    nargo new my_program
    ```

2. Navigate to the folder:

    ```bash copy
    cd my_program
    ```

3. Update the program with your favorite text editor to:

    ```rust copy filename="src/main.nr"
    fn main(x: pub u64, y: u64) {
        assert(x % y == 0);
    }
    ```

    This program allows one to prove that they know of a private factor `y` of a public integer `x`.

4. Run a small check to generate what you need:

    ```bash copy
    nargo check
    ```

5. We're almost there, change prove inputs to:

    ```toml copy filename="Prover.toml"
    x = "4611686014132420609"
    y = "2147483647"
    ```

6. Finally, we're ready to start proving:

    ```bash copy
    nargo prove
    ```

    Congratulations 🎉, you've made your first proof! Now we can verify it:

    ```bash copy
    nargo verify
    ```

You've now successfully written and proven a Noir program! Feel free to play around, for example, if you change `y` to `3` in `Prover.toml`, you'll get a prove error.

Once you're done, head over to [noir-lang.org](https://noir-lang.org/) and start learning about the language.
