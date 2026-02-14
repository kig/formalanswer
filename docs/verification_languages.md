# Programming Languages for Formal Verification

This document outlines the languages best suited for formal verification and how to link their verification artifacts to actual implementations.

## 1. Top-Tier Proof Assistants (The "Brain" Languages)

These languages are designed from the ground up for mathematical rigor and proof construction.

### A. Lean 4
*   **Best For**: Complex data invariants, structural logic, and verifying arithmetic.
*   **Linking Strategies**:
    *   **Direct Compilation**: Lean 4 compiles to efficient C code. You can write the implementation *as* the proof.
    *   **FFI**: Use Lean's foreign function interface to call verified Lean functions from Rust or C++.
    *   **Side-by-Side Sync**: Use tools like `impl-link` to assert that constants in your main app (Rust/TS) match your Lean definitions.

### B. Coq
*   **Best For**: Mission-critical kernels and compilers (e.g., CompCert).
*   **Linking Strategies**:
    *   **Extraction**: Coq can "extract" verified code into OCaml, Haskell, or Scheme.
    *   **Fiat Crypto**: A specialized Coq framework that generates verified C/Rust code for cryptographic primitives.

### C. F* (F-star)
*   **Best For**: Verifying low-level C code and security protocols.
*   **Linking Strategies**:
    *   **KreMLin**: A tool that translates verified F* code into readable, efficient C code (used heavily in the Firefox HTTPS stack).

---

## 2. Systems Languages with Verification Tooling

These are traditional languages augmented with powerful verification frameworks.

### A. Rust (The Modern Choice)
Rust's ownership model naturally lends itself to verification.
*   **Verus**: Allows writing proofs inside Rust code. It verifies that the Rust implementation adheres to the specified properties.
*   **Creusot**: Translates Rust code into Why3 logic for verification.
*   **Kani**: A model checker for Rust that checks for panics and user-defined assertions.
*   **Flux**: Refinement types for Rust, allowing you to specify numeric ranges and other properties directly in the type system.

### B. Ada (SPARK)
*   **Best For**: Aerospace, defense, and high-safety systems.
*   **Linking Strategy**: SPARK is a subset of Ada that is formally defined. The compiler itself performs the verification.

### C. C (Frama-C / SeL4)
*   **Frama-C**: A suite of tools for analysis and verification of C code using ACSL (ANSI/ISO C Specification Language).
*   **SeL4**: The most famous verified microkernel, which uses Isabelle/HOL to prove the correctness of its C implementation.

---

## 3. Modeling Languages (Behavioral Verification)

These are not "programming" languages in the traditional sense but are essential for verifying system behavior.

### A. TLA+ / PlusCal
*   **Best For**: Concurrency, distributed systems, and complex state transitions (deadlocks, race conditions).
*   **Linking Strategies**:
    *   **Trace Checking**: Run your implementation (Rust/Node) and emit a log of state transitions. Use TLC to verify that the trace is a valid execution of your TLA+ model.
    *   **PGo**: A tool that compiles PlusCal models into Go code (experimental).

---

## Summary of Linking Methods

| Method | Description | Best For |
| :--- | :--- | :--- |
| **Monolithic** | Writing implementation and proof in the same language (Verus, Lean). | Maximum correctness, higher effort. |
| **Extraction** | Generating code from a proof assistant (Coq, F*). | Core kernels, crypto. |
| **Cross-Verification** | Running implementation traces against a model (TLA+). | Distributed systems, protocols. |
| **Sync-Checking** | Using `impl-link` style scripts to ensure constants/enums match. | Pragmatic projects, UI constants. |
| **Refinement Types** | Encoding properties in types (Flux, LiquidHaskell). | Input validation, array bounds. |
