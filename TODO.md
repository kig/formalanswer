# ProofLoop Roadmap

## Completed
- [x] **Priority 1-13** (Reliability, Knowledge Reuse, Consistency, Tactics, Auto-Repair, Incremental Repair, Structural Alignment, Trace Explanation, Documentation, Parallel Verification, Rich CLI, Refactoring, Formal Specification)

## Priority 14: Business Model & Sustainability
**Goal:** Transition ProofLoop from a research tool to a production-ready reasoning engine with cost awareness and professional reporting.

- [x] **Usage Metering & Cost Tracking**
    - [x] Update `controller.py` to track tokens and estimated costs per reasoning task.
    - [x] Implement `src/utils/meter.py` to persist usage logs.
- [x] **Tiered Verification ("The Funnel")**
    - [x] Add `--tier` CLI flag (Standard, Pro, Enterprise).
    - [x] Update `controller.py` to adjust iterations and adversarial settings based on tier.
    - [x] Inject assurance context into Proposer prompts.
- [x] **Professional Logic Reports**
    - [x] Implement `src/ui/reporter.py` to generate standalone Markdown artifacts.
    - [x] Calculate "Formal Assurance Score" (FAS) based on tool coverage and proof depth.

## Future Work
- [ ] **Persistent Lean Server**: Implement a persistent Lean process to reduce verification latency.
- [ ] **Web UI**: Create a FastAPI/React frontend for visual interaction.
- [ ] **Cloud Deployment**: Containerize with Docker and deploy to Cloud Run.
