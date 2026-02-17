# Implementation Plan: Formal Extensions, Skills, and Discovery

This document tracks the implementation of the advanced FormalAnswer capabilities.

## Phase 1: Proof Discovery & Reuse (The "Librarian") [COMPLETE]
**Goal:** Enable the LLM to discover and use existing formal modules (`modules/`).

- [x] **1.1 Interface Extractor** (`src/utils/interface_extractor.py`)
- [x] **1.2 Module Indexer** (`src/library/indexer.py`)
- [x] **1.3 Librarian Agent** (`src/proposer/retriever.py`)
- [x] **1.4 Orchestrator Integration** (`src/main.py`)

## Phase 2: Formal Skills & API Integration (The "Contractor") [COMPLETE]
**Goal:** Safe execution of external tools via formal contracts.

- [x] **2.1 Skill Schema Definition** (`skills/examples/calculator.yaml`)
- [x] **2.2 Contract Generator** (`src/skills/contract_generator.py`)
- [x] **2.3 Skill Verifier / Injector** (`src/verifiers/skill_verifier.py`)

## Phase 3: Advanced Formal Extensions (The "Specialists") [NEXT]
**Goal:** Add specific high-level reasoning capabilities to augment the system.

- [ ] **3.1 Causal Verifier** (`src/verifiers/causal_verifier.py`)
    - **Purpose:** To verify claims like "A causes B" using Structural Causal Models (SCM).
    - **Logic:** Wrapper around `python_verifier.py` that validates Causal DAGs (using `causalnex` or `DoWhy` via JAX).
    - **Test:** Verify a simple causal query (e.g., Simpson's Paradox).
    
- [ ] **3.2 Skill Library Expansion** (`skills/`)
    - **Purpose:** Move beyond the calculator.
    - **Tasks:**
        - `skills/fs.yaml`: File system operations (read/write) with safety bounds (e.g., no overwriting without flag).
        - `skills/web.yaml`: Verified web search (e.g., result must match query keywords).
        
- [ ] **3.3 Refinement Mapper** (`src/verifiers/refinement_verifier.py`)
    - **Purpose:** To check if Generated Code (`impl`) actually matches the Formal Spec (`spec`).
    - **Approach:**
        - **Static:** Use an LLM-as-Verifier to critique code against TLA+ invariants.
        - **Dynamic:** Generate property-based tests (Hypothesis/CrossHair) from the TLA+ spec and run them against the Python code.

## Phase 4: User Experience & Information Architecture [IN PROGRESS]
- [x] **4.1 Prompt Preservation:** Save original prompts to `library/.../prompt.txt`.
- [x] **4.2 Answer Layout:** Refine the final output to be an "Executive Summary" rather than a raw dump of the reasoning process.