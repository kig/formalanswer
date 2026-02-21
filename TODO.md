## Priority 11: Architectural Refactoring
**Goal:** Improve code maintainability by splitting the monolithic `main.py` into specialized components.

- [x] **Extract UI Logic**
    - [x] Create `src/ui/cli.py` to handle `rich` output, argument parsing, and user interaction.
    - [x] Move `finalize_rap_battle` and feedback construction to UI module.
- [x] **Extract Controller Logic**
    - [x] Create `src/controller.py` to house the `FormalReasoningLoop` class.
    - [x] Ensure clear separation of concerns: Controller manages state/logic, UI manages display.

## Priority 12: Persistent Lean Server
**Goal:** Drastically reduce verification latency by keeping the Lean compiler process alive.

- [ ] **Implement `LeanServer`**
    - [ ] Create `src/verifiers/lean_server.py`.
    - [ ] Use `subprocess.Popen` to manage `lean --server` or a custom REPL wrapper.
    - [ ] Implement `verify(code)` that sends code to the persistent process and parses JSON/text response.
    - [ ] Handle timeouts and restarts.