# ProofLoop: Integration with Coding Agent CLIs (e.g., gemini-cli)

This document proposes an architecture for embedding **ProofLoop** as a "System 2" reasoning engine within a broader "System 1" coding agent CLI.

## 1. Critique & Design Challenges

Integrating a heavy formal verification loop into an interactive CLI presents distinct challenges.

### A. The Latency Mismatch
*   **Critique:** Coding CLIs rely on speed. Users expect "instant" code generation. ProofLoop's loop (Propose $	o$ Lean $	o$ TLA+ $	o$ Repair) takes minutes.
*   **Refinement:**
    *   **Explicit Invocation:** ProofLoop should not run on every prompt. It must be an explicitly invoked mode (e.g., `/verify` command) or a specialized tool called only for high-stakes logic.
    *   **Async Feedback:** The CLI should be able to continue working while ProofLoop verifies a specification in the background.

### B. Context & State Synchronization
*   **Critique:** The CLI holds the "Project Context" (open files, git status). ProofLoop needs this context to verify relevant logic, but dumping the entire repo into TLA+ is impossible.
*   **Refinement:**
    *   **Context Slicing:** The CLI must extract *only* the relevant logic (e.g., the `Auth` class) and pass it to ProofLoop.
    *   **Abstract Modeling:** ProofLoop should not verify the *implementation details* (lines of code) but the *logical model* derived from them.

### C. The "Output Gap"
*   **Critique:** ProofLoop produces `.tla` files and proofs. A CLI user wants *executable code* or a *diff*.
*   **Refinement:**
    *   **Code Synthesis Mode:** ProofLoop must output a "Golden Implementation" (Python/Rust/TS) derived from the verified spec, which the CLI then merges.
    *   **Verification Report:** For existing code, it outputs a "Pass/Fail" report with counterexamples.

---

## 2. Proposed Architecture: The "High-Assurance Sidecar"

ProofLoop operates as a local server or a specialized sub-process.

### The Interface (JSON-RPC)
The CLI communicates with ProofLoop via structured JSON.

**Request:**
```json
{
  "command": "verify_logic",
  "context": {
    "files": {"auth.py": "def login()..."},
    "intent": "Ensure no user can login without 2FA."
  },
  "constraints": ["Mode: Probabilistic", "Timeout: 300s"]
}
```

**Response:**
```json
{
  "status": "verified",
  "invariants": ["User.2FA = True"],
  "artifacts": {
    "spec": "auth.tla",
    "proof": "auth.lean"
  },
  "feedback": "The logic holds, but edge case X (database timeout) is unhandled."
}
```

## 3. Integration Scenarios

### Use Case A: The "Verified Refactor"
**User:** "Refactor the payment state machine. Make sure we never double-charge."
1.  **CLI (System 1):** Identifies `payment.py`. Recognizes high-stakes request.
2.  **CLI:** Calls `ProofLoop.synthesize_spec(payment.py, safety="no double charge")`.
3.  **ProofLoop:**
    *   Reverse-engineers a TLA+ spec from `payment.py`.
    *   Identifies the bug in the spec.
    *   Fixes the spec & verifies it.
    *   Returns the *Corrected Logic* description.
4.  **CLI:** Applies the logic to the code.

### Use Case B: The "Architectural Blueprint"
**User:** "Plan a distributed lock service."
1.  **CLI:** Calls `ProofLoop.design_system("distributed lock")`.
2.  **ProofLoop:**
    *   Retrieves `modules/safety/consensus.tla`.
    *   Verifies parameters for the user's scale.
    *   Returns a verified *Protocol Description*.
3.  **CLI:** Generates scaffolding code based on that protocol.

---

## 4. Implementation Plan

### Step 1: ProofLoop API Layer
Create `src/api.py` to expose `FormalReasoningLoop` as a callable service/library rather than just a CLI entry point.

### Step 2: The "Gemini-CLI" Tool Definition
Define a custom tool for the host CLI:

```typescript
// In gemini-cli tools definition
{
  name: "formal_verifier",
  description: "Verifies complex logic, distributed systems, or critical safety invariants. Use for high-stakes architecture tasks.",
  parameters: {
    object: "string", // The code or concept to verify
    safety_property: "string" // What must NEVER happen (invariant)
  }
}
```

### Step 3: The "Refinement" Loop
Implement a feedback handler where `ProofLoop`'s failure (e.g., "Counterexample found: State X") is fed back to the CLI's chat context so the user can see *exactly* why their logic is flawed.
