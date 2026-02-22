# ProofLoop Web Service Design

**Goal:** Provide a high-availability, zero-install Reasoning-as-a-Service (RaaS) platform for ProofLoop.

## 1. Core Architecture
The service follows a **Producer-Consumer** pattern to handle long-running reasoning tasks.

-   **Frontend:** React/Next.js dashboard for humans to view "Logic Rap Battles" and verify theorems.
-   **API (FastAPI):** Orchestrates requests, handles auth, and manages the task queue.
-   **Job Queue (Celery/Redis):** Asynchronous execution of the Agentic Loop.
-   **Execution Workers:** Sandboxed environments (gVisor/Docker) containing:
    -   Lean 4 (Persistent Server)
    -   TLA+ (TLC Model Checker)
    -   Python 3.10 (JAX/Z3)
-   **Artifact Store (S3/Minio):** Persists `logic_report.md` and proof source files.

## 2. API Endpoints (AX - Agent Experience)

### `POST /v1/reason`
**Mode:** Full Agency.
-   **Input:** Natural language query, Tier (Standard/Pro/Enterprise).
-   **Output:** `task_id`.
-   **Feature:** Supports **SSE (Server-Sent Events)** at `/v1/status/{task_id}` so agents can stream the "inner monologue" (Critiques, Repairs) in real-time.

### `POST /v1/verify`
**Mode:** Proof-Check Only.
-   **Input:** JSON containing code blocks (`lean`, `tla`, `python`).
-   **Output:** `VerificationResult` objects.
-   **Use Case:** Ideal for agents that have already written a proof and just need a "Formal Checksum".

### `GET /v1/library/search`
-   **Input:** Keywords.
-   **Output:** Relevant verified modules from the global library.

## 3. Red-Team Analysis (Adversarial Review)

### **Vulnerability 1: Arbitrary Code Execution (Python/Lean)**
-   **Attack:** A user submits a Python block that tries to access `/etc/passwd` or a Lean proof that uses `io` to execute shell commands.
-   **Critique:** ProofLoop's core value is running untrusted code from LLMs. Without strict isolation, the host is compromised.
-   **Mitigation:** 
    -   Workers must run in **Ephemeral Unprivileged Containers**.
    -   Use **gVisor** or **Firecracker** for strong kernel-level isolation.
    -   Network access disabled for workers during verification.

### **Vulnerability 2: Resource Exhaustion (Denial of Service)**
-   **Attack:** A user submits a TLA+ specification with an infinite state space, crashing the JVM or consuming all RAM.
-   **Critique:** Model checking is $O(exp(N))$. A single malicious spec could hang the entire worker pool.
-   **Mitigation:**
    -   Strict **Timeouts** (e.g., 60s for TLA+, 30s for Lean).
    -   **cgroups** to limit memory per task (e.g., 2GB per TLC run).
    -   Tier-based rate limiting (Enterprise gets priority).

### **Vulnerability 3: Prompt Injection in the Repair Loop**
-   **Attack:** A user inputs a query that tricks the internal "Red Team" LLM into approving a false proof.
-   **Critique:** If the "Judge" is compromised, the "Verified" status becomes meaningless.
-   **Mitigation:**
    -   The **Verification Result** must be mechanical. Even if the LLM Judge is tricked, the Lean compiler cannot be tricked. 
    -   The system must explicitly separate "LLM Judgement" from "Formal Compilation" in the FAS (Formal Assurance Score).

## 4. Optimal UX/AX Strategy
-   **Markdown Links:** API returns a permanent URL to the `logic_report.md` (e.g., `proofloop.ai/report/{hash}`).
-   **GitHub Integration:** A GitHub Action that calls `/v1/verify` on every PR containing formal specs.
-   **Proof Badges:** Scalable Vector Graphics (SVG) badges for READMEs: `[ProofLoop: Verified (FAS 95)]`.
