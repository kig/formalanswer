
## Priority 7: Trace Explanation (System 2 Debugging)
**Goal:** Help the LLM understand *why* TLA+ failed by translating raw counter-example traces into narrative explanations.

- [x] **Trace Extraction**
    - [x] Update `tla_verifier.py` to return the trace as a structured object or clean string separate from the error message.
- [x] **LLM Explainer**
    - [x] Add `explain_trace(trace, spec)` to `proposer/client.py`.
    - [x] Prompt: "Here is a TLA+ spec and a counter-example trace. Explain logically why the invariant was violated."
- [x] **Integration**
    - [x] Update `main.py` to call `explain_trace` when TLA+ fails with a trace.
    - [x] Append the explanation to the feedback loop.
