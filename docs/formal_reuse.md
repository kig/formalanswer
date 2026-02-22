# ProofLoop: Modular Reasoning & Subroutine Reuse

This document proposes a framework for creating and utilizing a library of "Trusted Formal Subroutines" to accelerate complex predictions.

## 1. Critique & Obstacles
Before implementing modular reuse, we must address critical logical and technical barriers.

### A. The "Context Window" Fallacy
*   **Objection:** Simply injecting full library files into the prompt will exhaust the LLM's context window and distract it with implementation details it doesn't need to modify.
*   **Refinement:** The system must implement **Interface Extraction**. The LLM should only see the *signatures* (Theorem statements, Function definitions, TLA+ Constants) of the library modules, not the proofs.

### B. The TLA+ Namespace Collision
*   **Objection:** TLA+ is strictly scoped. `EXTENDS` pulls everything into the global namespace, causing name collisions (e.g., two modules defining `Init`).
*   **Refinement:** We must strictly use TLA+ `INSTANCE` with parameterization. The LLM must be trained/prompted to instantiate modules rather than extend them (e.g., `Inner == INSTANCE CircuitBreaker WITH Threshold <- 10`).

### C. Verification Overhead
*   **Objection:** Re-verifying the entire dependency tree for every query is computationally wasteful.
*   **Refinement:**
    *   **Lean:** Use `lake` to pre-compile modules into `.ole` files.
    *   **TLA+:** Assume properties of the instantiated module are true (Trust-but-Verify later), or run composition proofs.

---

## 2. Refined Implementation Guide

### Phase 1: The "Interface" Retriever (Complexity: Medium)
Instead of catting files, we parse them to extract public interfaces.

*   **Logic:** `Interface(Module) = { Theorems, Definitions, Constants } - { Proofs, Local Variables }`
*   **Implementation:**
    1.  Create `src/utils/interface_parser.py`.
    2.  For Lean: Extract lines starting with `theorem`, `def`, `structure`.
    3.  For TLA+: Extract `CONSTANT`, `VARIABLE`, and labeled definitions.
    4.  **Cost:** ~2 days dev time.

### Phase 2: The Module Registry (Complexity: Low)
Structure the `library/` to distinguish between "Queries" (one-off) and "Modules" (reusable).

```text
proof_loop/
├── library/ (User Queries)
└── modules/ (Verified Components)
    ├── math/
    │   ├── probabilities.lean      (Complexity: Low)
    │   └── statistics.py           (Complexity: Low)
    └── safety/
        ├── circuit_breaker.tla     (Complexity: Medium - generic TLA+ is hard)
        └── consensus.tla           (Complexity: High)
```

### Phase 3: The Build System Linker (Complexity: High)
Dynamically link the user's query with the selected modules during verification.

*   **Lean 4:**
    *   **Method:** Add `modules/` to `lakefile.toml` paths dynamically.
    *   **Code:** `src/verifiers/lean_verifier.py` must append the module path to `LEAN_PATH`.
    *   **Complexity:** Medium.
*   **TLA+:**
    *   **Method:** Copy referenced `.tla` files from `modules/` to `work/` temp dir before running `tlc`.
    *   **Code:** `src/verifiers/tla_verifier.py` detects `INSTANCE` or `EXTENDS` and fetches files.
    *   **Complexity:** Low.
*   **Python/JAX:**
    *   **Method:** Add `modules/` to `PYTHONPATH`.
    *   **Complexity:** Very Low.

---

## 3. Workflow Example: "The Hedged Bet"

**User Query:** "Design a trading bot that hedges if volatility > 5."

1.  **Retrieval:** System scans `modules/` and finds `finance.volatility`.
2.  **Prompting:**
    *   System injects `finance.volatility` **Interface**:
        ```lean
        -- From modules/finance/volatility.lean
        def Volatility (prices : List Float) : Float
        theorem vol_non_negative : ∀ p, Volatility p ≥ 0
        ```
3.  **LLM Generation:**
    *   LLM writes: `import Modules.Finance.Volatility`
    *   LLM proves: `theorem safe_hedge : Volatility p > 5 → TriggerHedge` using `vol_non_negative`.
4.  **Verification:**
    *   `lean_verifier.py` links the pre-compiled module.
    *   Proof passes instantly.

## 4. Strategic Recommendation
Start by creating a manually curated `modules/` directory with **3 core primitives**:
1.  **Math:** `Gaussian.lean` (Normal distribution bounds).
2.  **Safety:** `RateLimiter.tla` (Generic token bucket).
3.  **Sim:** `MonteCarlo.py` (JAX loop template).

This provides immediate value ("Superhuman Speed") with minimal architectural overhaul.