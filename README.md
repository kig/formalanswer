# FormalAnswer - The Formal Reasoning Loop

**FormalAnswer** is a "System 2" orchestrator for Large Language Models that uses **Formal Methods** to mechanically verify natural language reasoning.

It implements a "Neural-Algebraic Mirror": the LLM proposes a logical argument, and FormalAnswer validates it using a suite of formal tools (Lean 4, TLA+, Z3) before accepting it.

## The Modern Reasoning Stack

This project uses the **"Lean 4 + TLA+"** stack, representing the 2026 gold standard for automated reasoning:

1.  **Lean 4 (The Universal Verifier):**
    *   Acts as the "Brain" and "Constraint Solver".
    *   Uses **Mathlib** and **Aesop** (Best-First Proof Search) to handle complex logical branching.
    *   Uses **SMT-style tactics** (`linarith`, `omega`) to verify arithmetic invariants and data consistency.
    *   Replaces the need for a separate Z3 script by doing SMT solving directly in the proof kernel.

2.  **TLA+ (The Safety Inspector):**
    *   Acts as the "Architect's Blueprint".
    *   Specializes in **Temporal Logic** to verify state transitions, concurrency, and deadlock freedom.
    *   Ensures the *process* of reasoning is safe over time.

3.  **Z3 (The Constraint Optimizer - Optional):**
    *   Acts as the "Heavy Lifter" for complex combinatorial problems.
    *   Used when the problem involves difficult optimization or constraint satisfaction tasks better suited for a dedicated solver.

## Installation

1.  **Prerequisites:**
    *   Python 3.10+
    *   [Lean 4](https://leanprover.github.io/lean4/doc/setup.shtml) (via `elan`)
    *   Java 11+ (for TLA+)

2.  **Setup:**
    ```bash
    # Install dependencies
    ./install_deps.sh
    
    # Initialize Lean environment (downloads Mathlib cache)
    cd work && lake exe cache get && cd ..
    ```

3.  **Environment Variables:**
    Create a `.env` file with your Google Gemini API Key:
    ```
    GOOGLE_API_KEY=your_key_here
    ```

## Agentic Development Process
When extending this project, follow this "Test-Driven Agentic" process:

1.  **Define the Interface:** If adding a new tool, define its `VerificationResult` contract in `common.py` first.
2.  **Mock the Proposer:** Use `src.proposer.client.Proposer._get_mock_response` to test the verification pipeline without spending API credits.
3.  **Verify the Verifier:** Create a simple "hello world" file (like `smoke_test.lean`) to prove the toolchain works *before* integrating it into the loop.
4.  **Iterate on Prompts:** Adjust `prompts.py` to enforce stricter schemas (like the "Shared Constants" rule) if the LLM struggles with consistency.

### Testing
Run unit tests before pushing any changes:
```bash
./pre-push.sh
```
This runs the `tests/` suite, checking code extraction and mock API interactions.

## Usage

Run a natural language query through the Formal Reasoning Loop:

```bash
./query.sh "Why is it a good idea to buy BTC today?"
```

### Advanced Usage (Backends)

You can specify different LLM backends using command-line flags:

**1. Gemini (Default)**
```bash
./query.sh "Question..." --backend gemini --model gemini-2.5-flash
```

**2. OpenAI**
```bash
export OPENAI_API_KEY=sk-...
./query.sh "Question..." --backend openai --model gpt-4o
```

**3. Ollama (Local)**
```bash
# Ensure Ollama is running (ollama serve)
./query.sh "Question..." --backend ollama --model llama3
```

**Full Options:**
*   `--backend`: `gemini` (default), `openai`, `ollama`
*   `--model`: Specific model name.
*   `--base-url`: Custom API endpoint (e.g., for vLLM or custom Ollama port).
*   `--api-key`: Override API key from CLI.

## Architecture

1.  **Proposer (LLM):** Generates a Rationale, TLA+ Spec, and Lean 4 Proof.
2.  **Verifier (Python):** 
    *   Compiles the TLA+ spec using `tla2tools.jar`.
    *   Compiles the Lean 4 proof using `lean`.
3.  **Feedback Loop:** If verification fails, the specific error logs (compiler messages, counter-examples) are fed back to the LLM for a "Repair" attempt.
4.  **Success:** Only when *both* verifiers pass is the reasoning accepted and summarized.

## Directory Structure

*   `src/`: Python source code.
*   `work/`: Temporary workspace for Lean/TLA+ files.
*   `library/`: Storage for successfully verified proofs.
*   `debug/`: Logs of each iteration (prompts, raw code, errors).

## Why FormalAnswer?

# Rationale & Shared Constants

The name **"FormalAnswer"** reflects the project's core mission: providing answers that are not just probable, but formally verified. It signifies the transition from opaque LLM reasoning to transparent, mathematically sound proofs. 

### Analysis of the Formal Argument
1.  **Invariants Proved**: 
    *   **Bounded Execution**: The TLA+ model proves that the system cannot enter a state where `attempts > MAX_ATTEMPTS`, ensuring the "Governor" always terminates.
    *   **Arithmetic Integrity**: The Lean 4 proof verifies that the transition function `n -> n+1` is logically sound within the bounds of our natural number constants.
2.  **Assumptions**: 
    *   The model assumes verification is a discrete, observable event.
    *   It assumes that the state "SUCCESS" is a terminal sink, meaning once a proof is verified, the truth value is stable.
3.  **Validity Conditions**: 
    *   This reasoning holds provided the underlying formal verifier (e.g., Lean or Coq) acts as a trusted oracle for the "VERIFYING" transition.

**Shared Constants:**
- `MAX_ATTEMPTS`: 5 (The maximum retry budget for the LLM to self-correct).
- `SUCCESS_STATE`: "SUCCESS".
- `EXHAUSTED_STATE`: "EXHAUSTED".

# TLA+ Specification (The Safety Inspector)

This specification models the FormalAnswer lifecycle, ensuring that the project's verification loop is temporally safe and respect the computational bounds.

```tla
---- MODULE temp ----
EXTENDS Naturals, TLC

\* Concrete values from Rationale
MaxAttempts == 5

VARIABLES attempts, status

Init == 
    /\ attempts = 0 
    /\ status = "IDLE"

\* Define the state transitions for the FormalAnswer Governor
Next == 
    \/ /\ status = "IDLE"
       /\ attempts < MaxAttempts
       /\ status' = "VERIFYING"
       /\ attempts' = attempts + 1
    \/ /\ status = "VERIFYING"
       /\ \/ status' = "SUCCESS"    \* Verification passed
          \/ (attempts < MaxAttempts /\ status' = "IDLE") \* Retry if possible
          \/ (attempts = MaxAttempts /\ status' = "EXHAUSTED") \* Budget spent
       /\ UNCHANGED attempts
    \/ /\ (status = "SUCCESS" \/ status = "EXHAUSTED") \* Terminal states
       /\ UNCHANGED <<attempts, status>>

\* Safety: The Governor never exceeds the allocated attempt budget.
SafetyInvariant == attempts <= MaxAttempts

\* Liveness: The system eventually reaches a terminal state.
Termination == <>(status = "SUCCESS" \/ status = "EXHAUSTED")

Spec == Init /\ [][Next]_<<attempts, status>>
====
```

# Lean 4 Proof (The Universal Verifier)

This proof verifies the logical and arithmetic consistency of the FormalAnswer budget mechanism, ensuring the "Generate-and-Verify" loop is mathematically sound.

```lean
import Mathlib
import Aesop

-- Set the concrete value for the formal budget
def MAX_ATTEMPTS : ℕ := 5

/-- 
  Theorem: Successive Safety.
  We prove that the 'increment' logic used in the TLA+ Next relation
  is arithmetically consistent with the safety invariant.
-/
theorem formalanswer_transition_consistent (n : ℕ) (h : n < MAX_ATTEMPTS) : 
  n + 1 <= MAX_ATTEMPTS := by
  -- linarith / omega handles the bounded arithmetic of Naturals
  omega

/-- 
  Theorem: State Exhaustion.
  We prove that once the attempt counter reaches the maximum, 
  it is logically impossible to satisfy the condition for another increment.
-/
theorem formalanswer_exhaustion_logic (n : ℕ) (h : n = MAX_ATTEMPTS) :
  ¬(n < MAX_ATTEMPTS) := by
  -- Aesop uses logical search to prove the contradiction
  aesop

/-- 
  Verification of the specific chosen constant.
-/
example : MAX_ATTEMPTS = 5 := rfl
```
