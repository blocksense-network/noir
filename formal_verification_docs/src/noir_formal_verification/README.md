# Noir FV overview

We at Blocksense have created a tool for verifying the correctness of code written in Noir. Building upon Microsoft's open-source verification framework Verus, we're able to prove functional correctness of constrained code, before it is ever ran! Noir FV introduces no run-time checks, but instead uses computer-aided theorem proving to statically verify that executable code will always satisfy some user-provided specifications for all possible inputs.  

Constrained Noir is not [Turing-complete](https://en.wikipedia.org/wiki/Turing_completeness) which simplifies proof writing. Noir’s deterministic nature eliminates the need for complex clauses which appear in other formal verification systems (like [invariants](https://viperproject.github.io/prusti-dev/user-guide/tour/loop_invariants.html?#loop-invariants) and [decrease clauses](https://verus-lang.github.io/verus/guide/reference-decreases.html?#the-decreases-measure)). Alongside the absence of a heap, we see Noir as a strong candidate for formal verification.


# This guide

Noir FV reuses Noir's syntax and type system to express specifications for executable code. Basic proficiency with the language is assumed by this guide. Noir's documentation can be found [here](https://noir-lang.org/docs).

Code specification needs to be written according to a number of predetermined rules, using clauses like `forall` and `requires`. Of course, you'll also need to grasp and conform to the required rigor for creating, what is effectively, a mathematical proof. It can be challenging to prove that a Noir function satisfies its postconditions (its ensures clauses) or that a call to a function satisfies the function’s preconditions (its requires clauses). 

Therefore, this guide’s tutorial will walk you through the various concepts and techniques, starting with relatively simple concepts (e.g., basic properties about integers), progressing to moderately difficult challenges, and eventually covering advanced topics like proofs about arrays using `forall` and `exists`.

All of these proofs are supported by an automated theorem prover (specifically, Z3, a satisfiability-modulo-theories solver, or “SMT solver” for short). The SMT solver will often be able to prove simple properties, such as basic properties about booleans or integer arithmetic, with no additional help from the programmer. However, more complex proofs often require effort from both the programmer and the SMT solver. Therefore, this guide will also help you understand the strengths and limitations of SMT solving, and give advice on how to fill in the parts of proofs that SMT solvers cannot handle automatically.
