# Blocksense Noir

At Blocksense, we’ve enhanced the Noir programming language and compiler with advanced tools designed to simplify and strengthen the creation of secure, verifiable circuits. Our current prototypes include:

  1. [Noir time-travel debugging](./noir_tracer) in the [CodeTracer](https://github.com/metacraft-labs/codetracer) environment;
   
  2. [formal verification](./noir_formal_verification) of Noir circuits, based on the Z3 SMT solver and the IR language of the [Verus project](https://github.com/verus-lang/verus); and
   
  3. [Noir compilation support](./noir_plonky2_backend) for the PLONKY2 proof system.
    
All of these developments are expected to reach a production-ready status in the future. We plan on merging them with the upstream codebase as soon as they are accepted by the Noir team.

The Blocksense Noir compiler follows the development of upstream Noir closely and it should be fully compatible.
