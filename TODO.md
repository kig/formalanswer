## Priority 6: Structural Alignment (Sim-to-Real Phase 2)
**Goal:** Ensure the Python simulation structurally mirrors the TLA+ specification's actions.

- [x] **Action Extraction**
    - [x] Update `consistency_checker.py` to extract action names from the TLA+ `Next` formula (e.g., `Next == \/ Relocate \/ ThreatShift` -> `['Relocate', 'ThreatShift']`).
    - [x] Update `consistency_checker.py` to extract function names from the Python script.
- [x] **Structure Comparison**
    - [x] Implement `check_structure(tla_content, python_content)` to warn if major TLA+ actions are missing corresponding Python functions.
    - [x] Integrate into `main.py`'s consistency check block.