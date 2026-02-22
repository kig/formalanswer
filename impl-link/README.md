# Linking formal proofs to implementations (HOWTO)

Use this **ProofLoop-style** loop to keep agent-written code honest:

1) A **formal oracle** (TLA+ and optionally Lean) computes *expected facts* for a deterministic fixture.
2) The **implementation** computes the same facts (usually via a `--json` mode).
3) A **harness test** runs both and diffs the results.

The key idea is: **formal proofs become executable oracles** over fixtures.

---

## 1) Define the “facts contract” (the bridge)

Pick a deterministic, inspectable output for your checker / subsystem and treat it as an API.

Good examples:
- `violations: string[]` (e.g. files exceeding a limit)
- `graph: { nodes: string[], edges: [string,string][] }`
- `budgets: { rawBytes: number, gzipBytes: number }`

Rules:
- Facts must be **pure** (no timestamps, random ids, absolute host paths).
- Facts must be **stable** (sorted arrays, normalized paths).
- Facts must be **small** (fixture-sized); keep it fast.

---

## 2) Implementation side (JS/TS)

Add a mode that prints the facts deterministically:

```bash
node scripts/janitor/my-check.js --root <fixture> --json
```

Recommendations:
- Output **only** JSON to stdout in `--json` mode.
- Still exit non-zero when `ok=false` (CI should fail), but always print facts.

---

## 3) Formal side (TLA+ harness)

Write a small harness spec that:

1. Defines the fixture inputs as constants (or uses the TLC working directory).
2. Computes expected facts as a pure expression.
3. Prints them in a single, greppable line.

Pattern:

```tla
---- MODULE MyHarness ----
EXTENDS TLC

\* Expected == ...

Init == Print("EXPECTED_FACTS=" \o ToString(Expected)) = Expected
Next == FALSE
Spec == Init /\ [][Next]_<<>>
====
```

Notes:
- Prefer printing a **TLA value** (sets/records) and parsing it.
- Keep output to a single line prefix, e.g. `EXPECTED_FACTS=`.

---

## 4) Optional: Lean “sanity lock”

Lean is best used to:
- prove tiny arithmetic/order lemmas used by the harness
- lock shared constants into the kernel (`MAX_LINES = 100`)

Lean usually won’t execute the JS implementation, but it prevents silent drift in spec constants.

---

## 5) Harness test (Jest)

Use `proofloop/impl-link/index.ts` helpers to:

1) locate the ProofLoop toolchain (the `work/` folder)
2) run TLC
3) extract expected facts from TLC stdout
4) create a temp fixture tree
5) run the implementation in `--json` mode
6) diff

Tip: keep each harness focused on one checker/subsystem and one small fixture.

