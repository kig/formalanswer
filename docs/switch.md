# Switch Logic: Discrete vs. Probabilistic Reasoning

The `ProofLoop` system relies on a "Neural-Algebraic Mirror" where the LLM (Proposer) must decide the appropriate formal modality. This logic is encoded in the `SYSTEM_PROMPT`.

## 1. Formal Decision Logic

The LLM selects the reasoning mode based on the structure of the input query $Q$:

### Mode A: Discrete/Deterministic Reasoning
**Predicate:** $IsDeterministic(Q) \iff 
eg ContainsUncertainty(Q) \land 
eg IsPredictive(Q)$
- **Target Invariant:** $S \in SafeStates$ (Binary safety).
- **Execution:**
    - **TLA+:** Proves exhaustive state-space coverage (e.g., "Deadlock is impossible").
    - **Lean:** Proves absolute equalities or inequalities (e.g., $N < Max$).

### Mode B: Probabilistic/Predictive Reasoning
**Predicate:** $IsProbabilistic(Q) \iff ContainsUncertainty(Q) \lor IsPredictive(Q)$
- **Target Invariant:** $P(S 
otin SafeStates) < \epsilon$ (Bounded risk).
- **Execution:**
    - **TLA+:** Proves threshold-based control logic (e.g., "If $P > \epsilon$, then $Action$").
    - **Lean:** Proves distribution properties or Bayesian updates (e.g., "The tail risk is bounded by $X$").

## 2. Decision Matrix in `SYSTEM_PROMPT`

The prompt forces this choice via the **Rationale & Shared Constants** section:

| Feature | Discrete Path | Probabilistic Path |
| :--- | :--- | :--- |
| **Invariant Type** | Mathematical Rule (e.g., $x + y = z$) | Probabilistic Rule (e.g., $P(Fail) < 0.05$) |
| **Constants** | Physical/Logical Limits | Risk Thresholds, Confidence Levels |
| **TLA+ Role** | State reachability | Logic of the response to uncertainty |
| **Lean Role** | Structural soundness | Mathematical bounds on distributions |

## 3. Identified Limitations & Lack of Features

While the prompt provides a framework, there are several "Cognitive Gaps" where the LLM might fail to make the correct decision:

### A. Lack of Explicit Mode Trigger
The prompt does not provide a keyword-based trigger (e.g., "If query mentions 'risk', use Probabilistic Mode"). It relies on the LLM's internal "System 1" to recognize the need for probability. This can lead to **Deterministic Overfitting**, where the LLM tries to prove a market crash is "impossible" rather than "unlikely."

### B. Missing "Bayesian Prior" Enforcement
The current system does not force the LLM to identify its **priors** vs. its **evidence** in the prompt. For probabilistic tasks, the choice between frequentist and Bayesian reasoning is left to the LLM's whim, which can lead to logically inconsistent proofs in Lean.

### C. The "Simulation Gap"
The LLM cannot execute Monte Carlo simulations during the `propose` phase. It must *estimate* probabilities. Without an integrated Python simulation verifier, the LLM might choose a "Probabilistic Mode" but provide "hallucinated" constants that pass Lean's arithmetic check but are empirically false.

### D. Complexity of Teleological Reasoning
The prompt asks for "Teleological Arguments" (purpose-driven), but does not provide a formal template for **Utility Functions**. The LLM may struggle to decide whether a goal is a "Discrete Liveness Property" (Eventually $Goal$) or a "Probabilistic Expected Utility" ($E[U] > Threshold$).

## 4. Proposed Fix: Explicit Branching
To ensure the LLM chooses correctly, the prompt should be updated with a **Mode Selection Header**:

```markdown
IF the query involves risk, prediction, or data: 
   USE PROBABILISTIC MODE (Invariant: P(Fail) < Epsilon)
ELSE:
   USE DISCRETE MODE (Invariant: State != Fail)
```

## 5. Technical Implementation
For the **Probabilistic Mode**, we rely on high-performance simulation to empirically verify bounds. See `recommendations.md` for a detailed analysis of why **JAX/NumPyro** is the chosen engine for this task.
