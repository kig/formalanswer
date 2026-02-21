
## Priority 9: Parallel Verification
**Goal:** Reduce latency by running independent verifiers (Lean, TLA+, Python) concurrently.

- [x] **Refactor `verify_blocks`**
    - [x] Create a `VerifierRunner` class or helper that accepts a list of tasks.
    - [x] Use `concurrent.futures.ThreadPoolExecutor` to run `verify_tla`, `verify_lean`, and `verify_python` in parallel.
    - [x] Handle result aggregation and error reporting thread-safely.

## Priority 10: Rich CLI Output
**Goal:** Improve user experience with structured logging and progress bars.

- [x] **Integrate `rich`**
    - [x] Add `rich` to `install_deps.sh`.
    - [x] Replace `print()` calls in `main.py` with `console.print()`, using panels for feedback and status spinners for long-running tasks.
