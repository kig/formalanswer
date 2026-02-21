# FormalAnswer - The Formal Reasoning Loop

**FormalAnswer** is a "System 2" orchestrator for Large Language Models that uses **Formal Methods** to mechanically verify natural language reasoning.

It implements a "Neural-Algebraic Mirror": the LLM proposes a logical argument, and FormalAnswer validates it using a suite of formal tools (Lean 4, TLA+, JAX) before accepting it.

| Feature | Traditional LLM (System 1) | FormalAnswer (System 2) |
| :--- | :--- | :--- |
| **Reliability** | "Sounds correct" (Probabilistic) | **Is correct** (Deterministic) |
| **Logic** | Hallucinates plausible steps | Mechanical Proof (Lean 4) |
| **Safety** | Guesses edge cases | Exhaustive Search (TLA+) |
| **Grounding** | Statistical patterns | Empirical Simulation (JAX/Z3) |
| **Result** | A confident guess | **A Verified Formal Answer** |

---

## 🎤 Logic Rap Battles: Adversarial Reasoning

FormalAnswer isn't just a solver; it's a fighter. Use the `--rap-battle` mode to watch two logical agents tear each other's arguments apart in rhyming verse before settling the score with a formal proof.

```bash
./query.sh "Who wins in a battle between a centralized and a decentralized oracle?" --rap-battle
```

**Why?** Because finding a flaw in a proof is easier when you're trying to win a roast. This mode uses **Adversarial Red-Teaming** to force the LLM to find its own "Sim-to-Real" gaps.

---

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

3.  **Python/JAX/Z3 (The Empirical Grounding):**
    *   Acts as the "Heavy Lifter" for simulations, optimization, and constraint solving.
    *   **Z3:** Used for complex constraint satisfaction problems (scheduling, resource allocation).
    *   **JAX/NumPy:** Used for high-performance Monte Carlo simulations and probabilistic modeling.
    *   **Role:** Bridges the gap between abstract formalisms and real-world data or messy physics.

## New Features (Feb 2026)

FormalAnswer has evolved into a self-healing, learning system:

*   **Auto-Repair (The "Hammer"):** If a proof fails due to a weak tactic (e.g., `simp`), the system automatically attempts stronger alternatives (`aesop`, `omega`, `linarith`) *before* asking the LLM to rewrite.
*   **Knowledge Reuse (RAG):** Successful proofs are indexed in a local "Knowledge Base". When facing a new problem, the system retrieves relevant TLA+ and Lean snippets to seed the context.
*   **Sim-to-Real Consistency:** A strict validator ensures that the Python simulation structurally mirrors the TLA+ specification (constants match, actions have corresponding functions).
*   **Trace Explanation:** If TLA+ finds a counter-example, the system uses an LLM pass to translate the raw state dump into a plain-English explanation of *why* the invariant failed.
*   **Incremental Repair:** Retries request patches only for the failing blocks, preserving successful proofs and saving tokens.
*   **Parallel Verification:** Independent verifiers (Lean, TLA+, Python) run concurrently to reduce latency.
*   **Rich CLI:** Structured, color-coded output with status spinners and panels for better user experience.

## Reasoning Modes

The system automatically selects (or can be forced to use via `--mode`) one of four reasoning strategies:

1.  **Discrete (`[MODE: DISCRETE]`):**
    *   **Use Case:** Logic puzzles, mathematical proofs, algorithm verification, and absolute safety claims.
    *   **Tools:** Relies heavily on **Lean 4** (for deductive correctness) and **TLA+** (for state transitions).
    *   **Goal:** 100% certainty based on axioms.

2.  **Probabilistic (`[MODE: PROBABILISTIC]`):**
    *   **Use Case:** Risk assessment, forecasting, simulation, and scenarios with uncertainty.
    *   **Tools:** Uses **Python/JAX** for Monte Carlo simulations or Bayesian inference. Formal proofs (Lean/TLA+) might be used for model bounds but not for the final answer.
    *   **Goal:** Expected value optimization or probability distribution estimation.

3.  **Hybrid (`[MODE: HYBRID]`):**
    *   **Use Case:** Systems reacting to uncertain environments (e.g., a self-driving car controller).
    *   **Tools:** **TLA+** verifies the control logic (safety), while **Python/Z3** models the environment (physics/constraints).
    *   **Goal:** "Safe control under uncertain conditions."

4.  **Factual (`[MODE: FACTUAL]`):**
    *   **Use Case:** Historical facts, definitions, or simple queries where formal modeling is overkill.
    *   **Tools:** Skips TLA+/Lean. Uses **Python/Z3** only if needed to verify logical consistency of the retrieved facts.
    *   **Goal:** Accurate retrieval and basic logical consistency.

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
./query.sh "Schedule 3 meetings for 5 people with overlapping 'No-Fly' zones, where Person A cannot be in a room with Person B."
```

### 💡 Example Gallery

*   **Distributed Systems:** "Design a deadlock-free distributed locking protocol for a multi-region database."
*   **Game Theory:** "Calculate the Nash Equilibrium for a 3-player version of Rock-Paper-Scissors with a customized payout matrix."
*   **Supply Chain:** "Prove that a 'Just-in-Time' inventory system is stable under a 10% daily variance in transit times."
*   **Optimization:** "Find the most cost-effective arrangement of solar panels on a non-convex roof, avoiding shadows from nearby trees."

**Factual Mode (No Formal Proofs):**
```bash
./query.sh "What is the capital of France?" --mode factual
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
./query.sh "Question..." --backend ollama --model llama3.2:latest
```

**Full Options:**
*   `--backend`: `gemini` (default), `openai`, `ollama`
*   `--model`: Specific model name.
*   `--base-url`: Custom API endpoint (e.g., for vLLM or custom Ollama port).
*   `--api-key`: Override API key from CLI.
*   `--mode`: Force a specific reasoning mode: `discrete`, `probabilistic`, `hybrid`, `factual`.
*   `--combat`: Enable Adversarial Combat Mode (Red Team review).
*   `--peer-review`: Enable Constructive Peer Review Mode.
*   `--rap-battle`: Enable Logic Rap Battle Mode.
*   `--construct-rap`: Construct rap lyrics from history (`CURRENT` or path).
*   `--prompt-file`: Load the prompt from a text file.
*   `--max-iterations`: Maximum number of reasoning iterations (default: 5).
*   `--verbose`: Show detailed verification errors in output.

## Integrations

FormalAnswer can be used as a backend tool for other AI agents (Claude, ChatGPT, Gemini).

*   **Claude Desktop (MCP):** Native integration via the Model Context Protocol.
*   **Gemini/OpenAI:** Function calling definitions provided.

See [docs/integrations.md](docs/integrations.md) for setup instructions.

## Verification Ecosystem

For more details on the tools and languages that integrate best with FormalAnswer, see:
*   [Programming Languages for Formal Verification](docs/verification_languages.md): A guide to selecting implementation languages and linking them to formal specs.
*   [Verified Refinement Guidelines](docs/refined_logic_guidelines.md): Generic best practices for linking formal logic to actual implementations.

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
