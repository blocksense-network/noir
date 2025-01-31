# Noir FV overview

Noir FV is a tool for verifying the correctness of code written in Noir. The primary goal is to verify full functional correctness of constrained code, building on ideas from existing verification frameworks like Verus, Boogie, VCC, Prusti, Creusot, Coq, and Isabelle/HOL. Verification is static: Noir FV introduces no run-time checks, but instead uses computer-aided theorem proving to statically verify that executable Noir code will always satisfy some user-provided specifications for all possible executions of the code.  

Constrained Noir is not [Turing-complete](https://en.wikipedia.org/wiki/Turing_completeness) which simplifies proof writing. Noir’s deterministic nature eliminates the need for invariants and decrease clauses, and the absence of a heap further streamlines formal verification. These characteristics make Noir a strong candidate for formal verification.


# This guide

This guide assumes a basic familiarity with Noir programming. Understanding Noir is beneficial for Noir FV, as it builds on Noir’s syntax and type system to express specifications alongside executable code. There is no separate specification language—specifications are written using Noir syntax and checked by Noir’s type checker.

Nevertheless, verifying the correctness of Noir code requires concepts and techniques beyond just writing ordinary executable Noir code. For example, Noir FV extends Noir’s syntax with new concepts for writing specifications, such as `forall`, `exists`, `requires`, and `ensures`. It can be challenging to prove that a Noir function satisfies its postconditions (its ensures clauses) or that a call to a function satisfies the function’s preconditions (its requires clauses). Therefore, this guide’s tutorial will walk you through the various concepts and techniques, starting with relatively simple concepts (e.g., basic properties about integers), progressing to moderately difficult challenges, and eventually covering advanced topics like proofs about arrays using `forall` and `exists`.

All of these proofs are supported by an automated theorem prover (specifically, Z3, a satisfiability-modulo-theories solver, or “SMT solver” for short). The SMT solver will often be able to prove simple properties, such as basic properties about booleans or integer arithmetic, with no additional help from the programmer. However, more complex proofs often require effort from both the programmer and the SMT solver. Therefore, this guide will also help you understand the strengths and limitations of SMT solving, and give advice on how to fill in the parts of proofs that SMT solvers cannot handle automatically.
