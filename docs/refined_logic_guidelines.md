# Verified Refinement: Guidelines for Implementation

These guidelines outline how to use ProofLoop to move from probabilistic coding to verified refinement, ensuring that implementation artifacts are direct shadows of proven logic.

## 1. Unified Invariant Anchoring
Every system-critical constant (buffer sizes, scale factors, protocol versions) must be defined in a formal specification first.

*   **Rule**: Do not hardcode "magic numbers" in implementation code.
*   **Workflow**:
    1.  Define the constant in Lean (`def MAX_SIZE : Nat := 1024`) or TLA+ (`MaxSize == 1024`).
    2.  Use an automation script (like `verify_sync.js`) to grep the implementation (Rust/TS/Go) and assert parity.
    3.  Hard-fail the build on any deviation.

## 2. Temporal Behavioral Modeling (Trace Checking)
For complex state machines (synchronization, consensus, multi-step auth), use TLA+ to define safe transitions.

*   **Workflow**:
    1.  Model the protocol in TLA+.
    2.  Instrument the implementation to emit a "Verification Trace" (a log of state changes).
    3.  Use `impl-link` to run the trace against the TLA+ model using TLC's trace-checking mode.
*   **Benefit**: Catches "Bugs of Omission" (states you forgot to handle) and race conditions.

## 3. Logic-to-Code Refinement
Complex algorithms should be refined from formal proofs.

*   **Workflow**:
    1.  Write the algorithm in Lean 4.
    2.  Prove its correctness properties (e.g., bounds safety, termination).
    3.  Manually transpile the logic into the target language (Rust/TS).
    4.  Tag the implementation: `// @refines SynapseSync.lean#apply_delta`.
    5.  Use property-based testing (`proptest`, `fast-check`) to assert that the implementation's output matches the formal logic's execution.

## 4. Eliminating "Unsafe" Intuition
Critical code paths (handling user input, network data, or raw pointers) must be gated by verification.

*   **Workflow**:
    1.  Identify "Critical Paths" (e.g., handlers for `Multipart`, `AuthUser`).
    2.  Require a corresponding Lean/TLA+ spec for these paths.
    3.  Incorporate a "Janitor" check that audits these paths for unverified `unwrap()` calls or complex logic branches lacking formal anchoring.

## 5. Formal-First Codegen
When using AI or automation to generate code:
1.  **Propose Logic**: Generate the formal spec first.
2.  **Verify Logic**: Run the verifier.
3.  **Generate Implementation**: Use the verified spec as the primary prompt context for generating the implementation.
4.  **Verify Parity**: Run sync-checking scripts.
