---
name: proof-loop
description: Formal reasoning and verification governor. Use this skill when you need to verify complex logic, distributed systems, or mathematical proofs using Lean 4, TLA+, and JAX.
---

# ProofLoop: Formal Reasoning Skill

When activated, you have access to the `ProofLoop` reasoning engine. This allows you to transform "guessed" code into "verified" logic.

## When to use this skill:
1. **Critical Safety Invariants**: When designing code that must never fail (e.g., "no double spending", "no deadlocks").
2. **Complex Algorithms**: When implementing non-trivial logic that is difficult to test exhaustively.
3. **Formal Verification**: When requested to provide a Lean 4 proof or a TLA+ specification.

## Procedural Guidance:
1. **Analyze the problem**: Identify the core logical invariants.
2. **Invoke ProofLoop**: Use the `pl` command to start a formal reasoning loop.
   - Example: `./pl "Verify the deadlock-freedom of this mutex implementation" --tier enterprise`
3. **Iterate on Proofs**: If ProofLoop reports a failure, use the provided compiler errors or TLA+ counter-examples to refine your implementation.
4. **Finalize**: Once ProofLoop provides a **Verified Formal Answer**, use that verified logic as the source of truth for your final code output.

## Tooling:
- `./pl`: The primary CLI entry point for reasoning tasks.
- `src/verifiers/`: contains specialized verifiers for Lean, TLA+, and Python.
- `library/`: Persistent store of verified logical modules.
