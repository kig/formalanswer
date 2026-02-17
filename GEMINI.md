# FormalAnswer

## 1. Mission & Philosophy
**FormalAnswer** operates as a "System 2" governor for Large Language Models. It operationalizes the **"Neural-Algebraic Mirror"** concept:
- **System 1 (LLM):** Generates high-entropy, intuitive "candidate solutions" (The Proposer).
- **System 2 (FormalAnswer Kernel):** Mechanically verifies these candidates against strict logical and temporal constraints (The Auditor).

**Goal:** Transform the LLM from a probabilistic token generator into a verifiable reasoning engine. An output is only accepted if it compiles and passes all formal checks.

## 2. The Modern Reasoning Stack
We utilize a simplified, high-power stack optimized for the 2026 landscape of automated reasoning.

### A. Lean 4 (The Universal Verifier)
*   **Role:** The "Brain" and "Constraint Solver".
*   **Responsibilities:**
    *   **Logic:** Verifies structural arguments using **Aesop** (Best-First Proof Search).
    *   **Arithmetic:** Verifies data consistency and invariants using SMT-style tactics (`linarith`, `omega`).
    *   **Replacement:** Replaces standalone Z3 scripts by integrating satisfiability checking directly into the proof kernel.
*   **Key Libraries:** `Mathlib`, `Aesop`.

### B. TLA+ (The Safety Inspector)
*   **Role:** The "Architect's Blueprint".
*   **Responsibilities:**
    *   **Temporal Logic:** Verifies state transitions, concurrency, deadlock freedom, and safety properties over time.
    *   **Process Modeling:** Ensures the *behavior* of the reasoning agent or system is safe.
*   **Tooling:** `tla2tools.jar` (TLC Model Checker).

### C. Z3 (The Heavy Lifter - Optional)
*   **Role:** The "Constraint Optimizer".
*   **Responsibilities:**
    *   **Combinatorial Search:** Handles complex constraint satisfaction problems (CSP) and large-scale optimization tasks that are cumbersome in Lean.
    *   **Model Finding:** Efficiently finds counter-examples for logical models.
*   **Tooling:** `z3-solver` (Python).

## 3. System Architecture
The system follows a strict **Agentic Control Loop**:

1.  **The Proposer (`src.proposer`):**
    *   **Multi-Backend Support:** Connects to **Gemini**, **OpenAI**, or **Ollama** (local).
    *   Interacts with the LLM to generate candidate proofs.
    *   Injects a "Unified Invariant Anchor" prompt, forcing the LLM to define shared constants across natural language, Lean, and TLA+.
    *   Parses the response into code blocks.

2.  **The Verifiers (`src.verifiers`):**
    *   **`lean_verifier.py`**: Wraps the `lake env lean` command. Checks for compilation errors and `sorry` (incomplete proofs).
    *   **`tla_verifier.py`**: Wraps the `java -cp ... tlc2.TLC` command. Checks for deadlocks and invariant violations.
    *   **`z3_verifier.py`**: Executes optional Python Z3 scripts to check satisfiability (`sat`/`unsat`).
    *   **`common.py`**: Defines standard `VerificationResult` objects.

3.  **The Orchestrator (`src.main`):**
    *   Manages the "Repair Loop".
    *   **Stateful Persistence:** Retains successful proofs from previous iterations to minimize regression.
    *   **Targeted Feedback:** Constructs specific error prompts ("Lean passed, but TLA+ failed with error X...") to guide the LLM's self-correction.

## 4. Development Standards & Quality Assurance
We adhere to a high degree of software quality to ensure reliability and testability.

### Code Quality
*   **Type Hinting:** All Python code must be fully type-hinted.
*   **Modularity:** Logic is separated into `proposer`, `verifiers`, and `library` modules.
*   **Error Handling:** Verifiers must gracefully handle missing tools, timeouts, and malformed output.

### Testing Strategy
*   **Integration Tests:** The `./query.sh` script serves as the primary end-to-end integration test.
*   **Smoke Tests:** `work/smoke_test.lean` verifies the Lean toolchain and Mathlib availability.
*   **Debug Traceability:** Every iteration logs raw prompts, responses, and verifier outputs to the `debug/` directory.

### Dependency Management
*   **Python:** Managed via `venv` and `pip` (standard library + `google-genai`, `python-dotenv`).
*   **Lean:** Managed via `lake` (`lakefile.toml`). Dependencies (Mathlib) are pinned and cached.
*   **Java:** TLA+ relies on a local `tla2tools.jar` managed by `setup_manager.py`.

## 5. Agentic Development Process
When extending this project, follow this "Test-Driven Agentic" process:

1.  **Define the Interface:** If adding a new tool, define its `VerificationResult` contract in `common.py` first.
2.  **Mock the Proposer:** Use `src.proposer.client.Proposer._get_mock_response` to test the verification pipeline without spending API credits.
3.  **Verify the Verifier:** Create a simple "hello world" file (like `smoke_test.lean`) to prove the toolchain works *before* integrating it into the loop.
4.  **Iterate on Prompts:** Adjust `prompts.py` to enforce stricter schemas (like the "Shared Constants" rule) if the LLM struggles with consistency.
5.  **Be logical:**  Before each code change, write a comment with a formal logical reasoning behind it and why it minimizes the amount of code to maintain in the long run while preserving the wanted functionality.

## 6. Usage
*   **Run Query:** `./query.sh "Your question here"`
*   **Clean Build:** `cd work && lake build`
*   **View Logs:** `cat debug/iteration_X_results.txt`
