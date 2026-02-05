# Implementation Plan: Formal Reasoning Loop (FRL)

This plan outlines the steps to build the "System 2" orchestrator described in `GEMINI.md`.

## Phase 1: Environment & Tooling Verification
**Goal:** Ensure all external formal verification tools are accessible to the Python environment.
- [x] **Check Dependencies:**
    - [x] Verify `z3` python package is installed. (Installed in `venv`)
    - [x] Verify `lean` (Lean 4) executable is in PATH. (Installed locally in `work/.elan`)
    - [x] Verify `java` is installed. (Installed locally in `work/jdk`)
    - [x] Download `tla2tools.jar`. (Done via `setup_manager.py`)
- [x] **Create Workspace:**
    - [x] Create a `work/` directory for intermediate files.

## Phase 2: The Auditor (Verifier Modules)
**Goal:** Create Python wrappers to invoke each tool and parse its output.
- [x] **Z3 Verifier (`src/verifiers/z3_verifier.py`):**
    - [x] Function `verify_z3` implemented.
- [x] **TLA+ Verifier (`src/verifiers/tla_verifier.py`):**
    - [x] Function `verify_tla` implemented.
- [x] **Lean Verifier (`src/verifiers/lean_verifier.py`):**
    - [x] Function `verify_lean` implemented.

## Phase 3: The Proposer (LLM Interface)
**Goal:** Abstract the LLM interaction to allow for "Repair Loops".
- [x] **Prompt Engineering (`src/proposer/prompts.py`):** (Logic embedded in Proposer for now)
- [x] **LLM Client (`src/proposer/client.py`):**
    - [x] Mock client implemented for testing.
    - [x] Code extraction logic implemented.

## Phase 4: The Orchestrator (Main Loop)
**Goal:** Tie it all together into the FRL.
- [x] **Main Logic (`src/main.py`):**
    - [x] Class `FormalReasoningLoop` implemented.
    - [x] Feedback loop and iteration logic verified.

## Phase 5: Verification (The "Structured Thinking" Test)
**Goal:** Prove the system works using `example.md`.
- [x] **Initial Run:** Verified that Z3 passes and the loop handles failures correctly.
- [x] **Full Pass:** Fully verified with local Java and Lean installations.

## Current Status
- [x] Phase 1 Complete
- [x] Phase 2 Complete
- [x] Phase 3 Complete
- [x] Phase 4 Complete
- [x] Phase 5 Complete
