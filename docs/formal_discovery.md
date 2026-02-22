# ProofLoop: Proof Discovery & Context Injection

This document details the **Prompt Engineering** and **Runtime Mechanics** required to let the LLM discover and reuse formal proofs from the library.

## 1. The "Select & Inject" Architecture
We cannot dump the entire library into the context. We use a **Two-Stage Retrieval Process**:

1.  **Stage A (Selection):** The "Librarian Agent" (a cheap LLM call or RAG) selects relevant modules based on the user query.
2.  **Stage B (Injection):** The "Proposer Agent" receives the *Formal Interfaces* (signatures only) of the selected modules in its prompt.

## 2. Phase 1: The Module Index (`index.json`)
The system maintains a lightweight index of available formal components.

```json
[
  {
    "id": "Math.Probabilities",
    "description": "Standard probability bounds (Markov, Chebyshev, Chernoff).",
    "keywords": ["probability", "statistics", "bound", "risk"],
    "path": "modules/math/probabilities.lean"
  },
  {
    "id": "Finance.Volatility",
    "description": "Volatility models and hedging predicates.",
    "keywords": ["finance", "market", "hedge", "crash"],
    "path": "modules/finance/volatility.lean"
  }
]
```

## 3. Phase 2: The Selection Prompt
**Input:** User Query + `index.json` (Descriptions only).
**Output:** List of Module IDs.

**Prompt:**
```text
You are the Librarian.
QUERY: "Design a circuit breaker for a crypto exchange crash."
AVAILABLE MODULES:
1. Math.Probabilities: Standard probability bounds...
2. Finance.Volatility: Volatility models...
3. Sys.Consensus: Raft/Paxos...

Select the modules required to formally verify this system.
OUTPUT: JSON ["Math.Probabilities", "Finance.Volatility"]
```

## 4. Phase 3: Interface Injection (The "Context")
Once modules are selected, we extract their **Public Interface** (ignoring private proofs).

**Extraction Logic (Lean 4 Example):**
*   **Include:** `theorem`, `def` (signature only), `structure`, `class`.
*   **Exclude:** `proof`, `where`, internal tactics.

**Constructed Prompt for Proposer:**
```text
SYSTEM: ... (Standard ProofLoop Prompt) ...

CONTEXT - AVAILABLE FORMAL TOOLS:
--------------------------------------------------
[MODULE: Math.Probabilities]
-- Use these theorems to prove bounds. Do not re-prove them.
theorem chebyshev_inequality (μ σ k : Real) (hσ : σ > 0) (hk : k > 0) :
  P(|X - μ| ≥ k * σ) ≤ 1 / k^2

[MODULE: Finance.Volatility]
def IsVolatile (p : PriceSeries) : Prop
theorem high_vol_implies_risk : ∀ p, IsVolatile p → Risk(p) > 0.8
--------------------------------------------------

TASK: "Design a circuit breaker..."
```

## 5. Implementation Guide

### A. `src/library/indexer.py` (Complexity: Low)
*   Scans `modules/`.
*   Reads the top-level docstring of each file for the "description".
*   Generates `library_index.json`.

### B. `src/proposer/retriever.py` (Complexity: Medium)
*   **Step 1:** Loads `library_index.json`.
*   **Step 2:** Calls LLM (cheap model) with the "Librarian Prompt".
*   **Step 3:** Returns list of paths.

### C. `src/utils/interface_extractor.py` (Complexity: Medium)
*   Reads a `.lean` or `.tla` file.
*   Uses regex (or Tree-sitter) to strip function bodies/proofs.
*   Returns the "Header" text.

### D. `src/main.py` Integration
*   Before calling `Proposer`, run `Retriever`.
*   Pass the extracted interfaces to `Proposer.propose(task, context=interfaces)`.

## 6. Example Interaction
**LLM Output:**
```lean
import Mathlib
import Modules.Finance.Volatility -- System automatically links this

theorem circuit_breaker_safety (p : PriceSeries) :
  IsVolatile p → CircuitBreaker.status = Halted := by
  intro h_vol
  -- The LLM uses the injected theorem 'high_vol_implies_risk'
  have h_risk := high_vol_implies_risk p h_vol
  ...
```
