# Implementation Plan: Formal Extensions, Skills, and Discovery

This document tracks the implementation of the advanced FormalAnswer capabilities.

## Phase 1: Proof Discovery & Reuse (The "Librarian")
**Goal:** Enable the LLM to discover and use existing formal modules (`modules/`).

- [ ] **1.1 Interface Extractor** (`src/utils/interface_extractor.py`)
    - **Logic:** Parse `.lean` and `.tla` files. Strip proof bodies (`by ...`, `proof ...`) and internal implementations. Return clear signatures.
    - **Test:** `tests/test_interface_extractor.py`
        - Input: A `.lean` file with `theorem t : P := by simp`.
        - Output: `theorem t : P` (without the proof).
- [ ] **1.2 Module Indexer** (`src/library/indexer.py`)
    - **Logic:** Scan `modules/`. Extract docstrings. Generate `library_index.json`.
    - **Test:** `tests/test_indexer.py`
        - Setup: Create dummy `modules/test/math.lean`.
        - Verify: JSON contains correct ID, description, and path.
- [ ] **1.3 Librarian Agent** (`src/proposer/retriever.py`)
    - **Logic:** Two-stage LLM call. 1. Query + Index -> Selection. 2. Selection + Extractor -> Context.
    - **Test:** `tests/test_retriever_mock.py`
        - Mock LLM selects "Math.Probabilities".
        - Verify: `proposer.propose` is called with the extracted interface injected.
- [ ] **1.4 Orchestrator Integration**
    - **Logic:** Update `src/main.py` to initialize `Retriever` and modify `Proposer` context.

## Phase 2: Formal Skills & API Integration (The "Contractor")
**Goal:** Safe execution of external tools via formal contracts.

- [ ] **2.1 Skill Schema Definition**
    - **Logic:** Define YAML format for Skills (Inputs, Preconditions, Postconditions).
    - **Test:** Create `skills/examples/calculator.yaml`.
- [ ] **2.2 Contract Generator** (`src/skills/contract_generator.py`)
    - **Logic:** YAML -> Lean Axioms + TLA+ Operators.
    - **Test:** `tests/test_contract_gen.py`
        - Verify generated Lean code compiles.
- [ ] **2.3 Skill Verifier / Injector** (`src/verifiers/skill_verifier.py`)
    - **Logic:** Parse generated Python code. Inject `assert` statements based on YAML constraints.
    - **Test:** `tests/test_skill_verifier.py`
        - Input: Python code calling `calc.add(a, b)`.
        - Output: Code with `assert a > 0` (if specified in YAML).

## Phase 3: Advanced Formal Extensions (The "Specialists")
**Goal:** Add specific high-level reasoning capabilities.

- [ ] **3.1 Causal Verifier** (`src/verifiers/causal_verifier.py`)
    - **Logic:** Wrapper around `python_verifier.py` that validates Causal DAGs (using `causalnex` or similar via JAX).
    - **Test:** Verify a simple causal query ("Does X cause Y?").
- [ ] **3.2 Game Theory Verifier**
    - **Logic:** Optimization script (Z3/JAX) to find Nash Equilibria.
    - **Test:** Verify a generic "Prisoner's Dilemma" setup.

## Execution Order
1.  **Commit Documentation.**
2.  **Implement Phase 1.1 & 1.2** (Core Infrastructure).
3.  **Implement Phase 1.3 & 1.4** (Agent Integration).
4.  **Implement Phase 2** (Skills).
