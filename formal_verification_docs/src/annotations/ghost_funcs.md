# Ghost Functions
In many formal verification systems, there is a strict separation between code for mathematical proofs and for execution.
Some systems, like **Verus**, introduce explicit "ghost functions"—special functions used purely for proofs and omitted from compiled code.
Others, like **Prusti**, allow calling regular functions inside specifications without requiring them to be ghost functions.

**Noir FV does not have ghost functions**. Instead, we prioritize executable code while ensuring that functions used in specifications are **pure**—they must have no side effects and always return the same output for the same input.

The traditional approach of using ghost functions, proofs, and lemmas has its benefits, especially for reasoning about complex systems like distributed systems.
However, it also has drawbacks, such as reduced code reusability and increased mathematical complexity, which can make the verification process less user-friendly.

With Noir FV, we take a different path—focusing on executable code and usability.
Our approach simplifies formal verification, making it more intuitive without compromising rigorous correctness guarantees.