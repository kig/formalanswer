## Priority 4: Smart Tactics (The "Hammer")
**Goal:** Reduce trivial failures by equipping the LLM with robust, high-power tactics.

- [x] **Prompt Engineering for Tactics**
    - [x] Update `prompts.py` to explicitly recommend `aesop` for logic and `omega`/`linarith` for arithmetic.
    - [x] Add a "Troubleshooting Guide" in the system prompt for common errors (e.g., "If `simp` fails, try `unfold` then `ring`").
- [x] **Auto-Repair Script (Python Middleware)**
    - [x] Create `src/verifiers/auto_repair.py`.
    - [x] If Lean fails with "tactic failed", try substituting the tactic with a stronger one (e.g., replace `simp` with `aesop`) and re-run verification *locally* before asking the LLM.

## Priority 5: Interactive/Incremental Repair
**Goal:** Reduce token usage and context window bloat.

- [ ] **Diff-Based Repair**
    - [ ] Update `client.py` to ask the LLM for *patches* (e.g., "Replace lines 10-15 with...") instead of full files on retry.
    - [ ] Implement a simple patch applier in `main.py`.