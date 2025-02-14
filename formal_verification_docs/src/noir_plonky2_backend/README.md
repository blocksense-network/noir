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
