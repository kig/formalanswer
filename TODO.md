# FormalAnswer Roadmap

## Completed
- [x] **Priority 1-11** (Reliability, Knowledge Reuse, Consistency, Tactics, Auto-Repair, Incremental Repair, Structural Alignment, Trace Explanation, Documentation, Parallel Verification, Rich CLI, Refactoring)

## Priority 13: Formal Codebase Specification (The "Self-Reference" Phase)
**Goal:** Create a mathematically verified design of FormalAnswer itself to ensure maintainability and correctness.

- [x] **Formalize Core Orchestration (TLA+)**
    - [x] Create `spec/FormalAnswerLoop.tla` to model the state transitions in `controller.py`.
    - [x] Define Invariants: `Safety_VerificationSoundness` (Results must match verifier status), `Liveness_Termination` (System must reach terminal state).
    - [x] Verify with TLC.
- [x] **Formalize Parallel Runner (TLA+)**
    - [x] Create `spec/ParallelRunner.tla` to model the concurrency in `runner.py`.
    - [x] Prove result-index mapping consistency.
- [x] **Formalize Retrieval Logic (Lean 4)**
    - [x] Create `spec/RetrieverLogic.lean` to model the relationship between the Indexer and Retriever.
    - [x] Prove that retrieved context is strictly a subset of the local library.

## Future Work
- [ ] **Persistent Lean Server**: Implement a persistent Lean process to reduce verification latency.
- [ ] **Web UI**: Create a FastAPI/React frontend for visual interaction.
- [ ] **Cloud Deployment**: Containerize with Docker and deploy to Cloud Run.