# ProofLoop - The Formal Reasoning Loop

**ProofLoop** is a "System 2" orchestrator for Large Language Models that uses **Formal Methods** to mechanically verify natural language reasoning. 

It implements a "Neural-Algebraic Mirror": the LLM proposes a logical argument, and ProofLoop validates it using a suite of formal tools (Lean 4, TLA+, JAX) before accepting it.

| Feature | Traditional LLM (System 1) | ProofLoop (System 2) |
| :--- | :--- | :--- |
| **Reliability** | "Sounds correct" (Probabilistic) | **Is correct** (Deterministic) |
| **Logic** | Hallucinates plausible steps | Mechanical Proof (Lean 4) |
| **Safety** | Guesses edge cases | Exhaustive Search (TLA+) |
| **Grounding** | Statistical patterns | Empirical Simulation (JAX/Z3) |
| **Result** | A confident guess | **A Verified Formal Answer** |



---

## 🧠 How It Works: The Modern Reasoning Stack

ProofLoop acts as an automated judge and self-healing loop. If verification fails, specific error logs (compiler messages, counter-examples) are fed back to the LLM for a targeted "Repair" attempt. 



This project represents the 2026 gold standard for automated reasoning, utilizing a tri-pillar verification stack:

1. **Lean 4 (The Universal Verifier):** The "Constraint Solver." Uses SMT-style tactics (`linarith`, `omega`) and best-first proof search (`aesop`) to handle complex logical branching.
2. **TLA+ (The Safety Inspector):** The "Architect's Blueprint." Specializes in Temporal Logic to verify state transitions, concurrency, and deadlock freedom over time.
3. **Python/JAX/Z3 (The Empirical Grounding):** The "Heavy Lifter." Bridges abstract formalisms to real-world data using Z3 for constraint satisfaction and JAX for high-performance Monte Carlo simulations.

---

## 🚀 Showcase & Capabilities

ProofLoop automatically selects the best reasoning mode for your query (Discrete, Probabilistic, Hybrid, or Factual). Explore our [**examples/**](examples/) directory to see it in action:

* [**📊 VC Investment Logic**](examples/README.md): Complex probabilistic evaluation of AI startups verified via TLA+, Lean 4, and JAX.
* **💡 Distributed Systems:** "Design a deadlock-free distributed locking protocol for a multi-region database."
* **💡 Game Theory:** "Calculate the Nash Equilibrium for a 3-player version of Rock-Paper-Scissors."

### 🎤 Adversarial Reasoning (Logic Rap Battles)
ProofLoop isn't just a solver; it's a fighter. Finding a flaw in a proof is easier when you're trying to win a roast. Use `--rap-battle` to watch two logical agents tear each other's arguments apart in rhyming verse, using **Adversarial Red-Teaming** to find "Sim-to-Real" gaps before settling the score with a formal proof.

```bash
./query.sh "Who wins in a battle between a centralized and decentralized oracle?" --rap-battle

```

🎧 [Listen to an Adversarial Reasoning Loop here!](https://fhtr.org/music/?m=https://raw.githubusercontent.com/kig/formalanswer/main/examples/proofloop_rap_battle.mp3)

---

## ✨ New in 2026: Self-Healing Systems

ProofLoop has evolved into an agentic, learning system:

* **Auto-Repair & Incremental Patching:** Automatically attempts stronger proof tactics (`aesop`, `omega`) before asking the LLM to rewrite. When retrying, it only requests patches for failing blocks to save tokens.
* **Knowledge Reuse (RAG):** Successful proofs are indexed locally to seed context for future, similar problems.
* **Trace Explanation:** TLA+ counter-examples are automatically translated into plain-English explanations of *why* an invariant failed.

---

## 💻 Getting Started

### Installation

1. Ensure you have **Python 3.10+**, **Java 11+** (for TLA+), and **[Lean 4](https://leanprover.github.io/lean4/doc/setup.shtml)** installed.
2. Clone and setup:
```bash
./install_deps.sh
cd work && lake exe cache get && cd ..

```


3. Set your API Key in a `.env` file: `GOOGLE_API_KEY=your_key_here` (Gemini is the default backend).

### Usage

Run a natural language query through the loop:

```bash
./query.sh "Schedule 3 meetings for 5 people with overlapping 'No-Fly' zones..."

```

**Advanced CLI Options:**
ProofLoop supports multiple backends (`gemini`, `openai`, `ollama`).

```bash
./query.sh "Question..." --backend openai --model gpt-4o --mode discrete

```

*(Run `./query.sh --help` for full options, including `--combat`, `--peer-review`, and custom backend configurations.)*

---

## 🤯 Meta-Example: ProofLoop Verifying Itself

To prove the power of this system, we asked ProofLoop to formally verify its own internal retry loop. It generated the rationale, bounded the execution with TLA+, and mathematically proved the logic in Lean 4.

<details>
<summary><b>Click to view the TLA+ Spec and Lean 4 Proof of ProofLoop's Governor</b></summary>

**Shared Constants Identified:** `MAX_ATTEMPTS = 5`

**TLA+ Specification (Safety Inspector)**
Ensures the verification loop is temporally safe and terminates.

```tla
---- MODULE temp ----
EXTENDS Naturals, TLC
MaxAttempts == 5
VARIABLES attempts, status

Init == attempts = 0 /\ status = "IDLE"

Next == 
    \/ /\ status = "IDLE"
       /\ attempts < MaxAttempts
       /\ status' = "VERIFYING"
       /\ attempts' = attempts + 1
    \/ /\ status = "VERIFYING"
       /\ \/ status' = "SUCCESS"    
          \/ (attempts < MaxAttempts /\ status' = "IDLE") 
          \/ (attempts = MaxAttempts /\ status' = "EXHAUSTED") 
       /\ UNCHANGED attempts
    \/ /\ (status = "SUCCESS" \/ status = "EXHAUSTED") 
       /\ UNCHANGED <<attempts, status>>

SafetyInvariant == attempts <= MaxAttempts
Termination == <>(status = "SUCCESS" \/ status = "EXHAUSTED")
Spec == Init /\ [][Next]_<<attempts, status>>
====

```

**Lean 4 Proof (Universal Verifier)**
Verifies the logical and arithmetic consistency of the attempt budget.

```lean
import Mathlib
import Aesop

def MAX_ATTEMPTS : ℕ := 5

theorem proofloop_transition_consistent (n : ℕ) (h : n < MAX_ATTEMPTS) : 
  n + 1 <= MAX_ATTEMPTS := by
  omega

theorem proofloop_exhaustion_logic (n : ℕ) (h : n = MAX_ATTEMPTS) :
  ¬(n < MAX_ATTEMPTS) := by
  aesop

```

</details>

---

*For integration guides (Claude MCP, Gemini Function Calling) see [docs/integrations.md](docs/integrations.md).*


