# FormalAnswer: API Integration & Formal Skills (MCP)

This document proposes a "Formal Skill Adapter" architecture to safely integrate external API calls (e.g., Model Context Protocol - MCP) into the verified reasoning loop.

## 1. Critique & Obstacles
Integrating external APIs breaks the closed-world assumption of formal verification.

### A. Non-Determinism & Side Effects
*   **Objection:** API calls (e.g., `stripe.charge()`) have side effects and return non-deterministic values (e.g., `500 Error`). Formal tools like Lean/TLA+ require pure functions and exhaustive state coverage.
*   **Refinement:** We must treat API calls as **Non-Deterministic Oracles** modeled by **Contracts**.
    *   **TLA+:** Models the API as a process that *eventually* returns a value in a known set (`Result \in {Success, Failure}`).
    *   **Lean:** Axiomatizes the API function with a `constant` and assumes its Postconditions hold if Preconditions are met.

### B. Latency & Verification Timeout
*   **Objection:** Waiting for real API calls during the *verification* phase (e.g., inside a TLA+ model checker) is impossible.
*   **Refinement:**
    *   **Verification Phase:** Use **Mock Contracts** (Pure definitions).
    *   **Execution Phase:** Use the **Real API** via Python/JAX.

### C. The "Hallucinated Parameter" Risk
*   **Objection:** The LLM might generate valid code that calls an API with invalid parameters (e.g., `transfer(-100)`).
*   **Refinement:** The Formal System must verify the **Preconditions** of the API call *before* generation.

---

## 2. Refined Implementation Guide

### Phase 1: The "Formal Skill" Definition (Complexity: Medium)
Define a schema for "Verified Skills" that includes formal specs.

```yaml
# skills/stripe_charge.yaml
name: stripe_charge
inputs:
  amount: Nat
  currency: String
preconditions:
  - amount > 0
  - currency \in {"USD", "EUR"}
postconditions:
  - result.status \in {"success", "declined"}
  - result.amount == amount
```

### Phase 2: The Contract Generator (Complexity: High)
*   **Logic:** Transform the YAML schema into TLA+ and Lean definitions.
    *   **Lean:**
        ```lean
        constant stripe_charge (amount : Nat) (currency : String) : IO Result
        axiom charge_post : ∀ a c, amount > 0 → (stripe_charge a c).status ∈ {"success", "declined"}
        ```
    *   **TLA+:**
        ```tla
        CheckCharge(a, c) == a > 0 /\ c \in {"USD", "EUR"}
        ChargeOutcome \in {"success", "declined"}
        ```
*   **Implementation:** `src/skills/contract_generator.py`.

### Phase 3: The "Skill Verifier" (Complexity: High)
*   **Constraint:** The LLM's generated Python script must include runtime assertions derived from the formal contract.
*   **Implementation:** `src/verifiers/skill_verifier.py`.
    *   Parses the Python code.
    *   Injects `assert precondition(x)` before every API call.
    *   Injects `assert postcondition(y)` after every result.

### Phase 4: MCP Integration (Complexity: Medium)
*   **Constraint:** Map MCP (Model Context Protocol) tools to this Formal Schema.
*   **Method:**
    1.  Query MCP server for `list_tools`.
    2.  LLM generates the `formal_spec` (YAML) for the selected tool.
    3.  System verifies the spec (syntactically).
    4.  System generates the formal contracts (Phase 2).

---

## 3. Workflow Example: "Verified Stripe Refund"

**User Query:** "Refund transaction tx_123 if the amount > $50."

1.  **Skill Selection:** System identifies `stripe.refund`.
2.  **Contract Loading:** System loads `skills/stripe_refund.yaml` -> Generates `Stripe.lean` and `Stripe.tla`.
3.  **LLM Generation:**
    *   **Rationale:** "I will verify amount > 50 before calling refund."
    *   **Lean:** Proves `amount > 50` implies `refund_precondition` holds.
    *   **Python:**
        ```python
        if amount > 50:
            # Verified Precondition
            stripe.refund("tx_123")
        ```
4.  **Verification:**
    *   Lean confirms the logic is sound relative to the Axiom.
    *   TLA+ confirms that even if refund fails, the system state remains safe (e.g., logs error).

## 4. Strategic Recommendation
Start with **Phase 1 & 2**:
1.  Define a simple schema for **Internal Skills** (e.g., `fs.read_file`, `calculator`).
2.  Build the `contract_generator.py` to produce valid Lean/TLA+ headers from these schemas.
3.  This enables "Tool Use" without breaking the formal proofs.
