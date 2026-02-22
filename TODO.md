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

## Priority 15: Dogfooding & Ecosystem
**Goal:** Use ProofLoop to develop itself and integrate deeply with the agentic CLI ecosystem.

- [x] **Gemini-CLI Backend (Dogfooding)**: Use `gemini-cli` as a proposer backend to leverage existing auth and code-assistance subscriptions.
- [x] **CLI Ergonomics**: Added `pl` shorthand for rapid invocation.
- [x] **ProofLoop as a Gemini-CLI Skill**: Package ProofLoop so it can be installed via `gemini skills install`.

## Priority 16: Web UI & API (Reasoning-as-a-Service)
**Goal:** Transition to a cloud-native backend to support zero-install agentic reasoning.

- [ ] **FastAPI Scaffold**: Implement the core REST API in `src/api/server.py`.
- [ ] **Job Queue**: Setup Celery and Redis for handling long-running `FormalReasoningLoop` tasks.
- [ ] **Sandboxed Workers**: Create a Dockerfile for isolated verification (Lean/TLA+/Python).
- [ ] **React Frontend**: Develop a dashboard for visualizing logic battles and verification results.

## Priority 17: Advanced Reasoning Modalities
**Goal:** Enhance the Proposer's logical depth via specialized prompting strategies.

- [x] **Core Strategy Injection**: Integrated "Lifting/Landing", "Top-Down Refinement", and "Two-Sided Proofs" into `SYSTEM_PROMPT`.
- [ ] **Strategy-Specific Mode**: Add `--strategy` flag to switch between different meta-reasoning patterns.
- [ ] **Adaptive Precision Logic**: Dynamically adjust `max_iterations` or verifier timeouts based on the importance of sub-modules.

## Priority 18: Deep Reasoning Insights
**Goal:** Transition from "syntax verification" to "system property verification".

- [ ] **Adversarial Proof Battles**: Update `critique` prompts to force the Red Team to generate compilable TLA+/Lean counter-proofs.
- [ ] **Conservative Abstraction Proofs**: Implement guidance for proving that an abstract model is a "Safe-Heavy" approximation of reality.
- [ ] **Cross-Tool Anchoring**: Enable "Unified Invariants" where Z3 results (constants) are injected into TLA+ specifications.
- [ ] **Formal Resource Optimization**: Use Z3 to prove the necessity of Adaptive Precision under finite token budgets.

## Future Work
- [x] **Persistent Lean Server**: Implement a persistent Lean process to reduce verification latency.
- [ ] **Web UI**: Create a FastAPI/React frontend for visual interaction.
- [ ] **Cloud Deployment**: Containerize with Docker and deploy to Cloud Run.
