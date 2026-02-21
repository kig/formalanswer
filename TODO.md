# FormalAnswer Improvement Plan

## Priority 1: Reliability & Feedback (The "Self-Correcting Loop")
**Goal:** Drastically increase the probability that the LLM can fix its own errors by providing surgical, structured feedback from the formal tools.

- [x] **Enhance Lean 4 Error Parsing**
    - [x] Analyze raw `lake` output for common error patterns.
    - [x] Implement regex extraction for Line Number, Error Type, and Message.
    - [x] Format feedback to point specifically at the failing line in the prompt.
- [x] **Enhance TLA+ Error Parsing**
    - [x] Analyze `tla2tools` output (Parser errors vs. Model Checking errors).
    - [x] Extract "Error location" (line/column) and "Counter-example trace" (state transitions).
    - [x] Format feedback to distinguish between "Syntax Error" (fix code) and "Invariant Violation" (fix logic).
- [x] **Unified Feedback Injection**
    - [x] Update `main.py` to aggregate structured errors.
    - [x] Update `proposer/client.py` to present these errors clearly (e.g., "Fix Line 10: ...").

## Priority 2: Knowledge Reuse (The "Library")
**Goal:** Stop starting from scratch. Allow the system to learn from previous successful proofs.

- [ ] **Simple Proof Indexing**
    - [ ] Create a lightweight indexer that scans `library/` for `SUCCESS` states.
    - [ ] Extract `Shared Constants` and `Theorems` into a `knowledge_base.json`.
- [ ] **Context Retrieval (RAG)**
    - [ ] Update `Retriever` to search the local `knowledge_base.json`.
    - [ ] Inject relevant past TLA+/Lean snippets into the System Prompt as "Reference Patterns".

## Priority 3: Consistency Checks (The "Sim-to-Real" Bridge)
**Goal:** Ensure the Python simulation actually matches the Formal Spec.

- [ ] **Constant Consistency Validator**
    - [ ] Implement a regex-based extractor to find `CONSTANTS` in TLA+ and global vars in Python.
    - [ ] specific check: Warning if `MaxRetries = 5` in TLA+ but `MAX_RETRIES = 3` in Python.
