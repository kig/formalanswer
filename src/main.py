from src.verifiers.tla_verifier import verify_tla
from src.verifiers.lean_verifier import verify_lean
from src.verifiers.z3_verifier import verify_z3
from src.proposer.client import Proposer
from src.library_manager import LibraryManager
from src.verifiers.common import VerificationResult
import os

class FormalReasoningLoop:
    def __init__(self, max_iterations=5):
        self.max_iterations = max_iterations
        self.proposer = Proposer()
        self.library = LibraryManager()
        self.best_blocks = {"tla": None, "lean": None, "z3": None}
        if not os.path.exists("debug"):
            os.makedirs("debug")

    def run(self, task):
        feedback = None
        for i in range(self.max_iterations):
            print(f"\n--- Iteration {i+1} ---")
            
            # Call LLM
            response = self.proposer.propose(task, feedback)
            
            # Save raw response for debugging
            with open(f"debug/iteration_{i+1}_raw.txt", "w") as f:
                f.write(response)
            
            current_blocks = self.proposer.extract_code(response)
            
            # --- Stateful Memory & Completeness Check ---
            for kind in ["tla", "lean", "z3"]:
                if not current_blocks.get(kind) and self.best_blocks[kind]:
                    print(f"[{kind.upper()}] restoring previously successful block.")
                    current_blocks[kind] = self.best_blocks[kind]

            results = {}
            all_pass = True
            failing_tools = []
            
            # --- Verification ---

            # TLA+
            if current_blocks.get("tla"):
                if current_blocks["tla"] == self.best_blocks["tla"]:
                    print("[TLA+] Skipping re-verification (identical to last success).")
                    results["tla"] = VerificationResult(True, "Identical to last success")
                else:
                    print("\n[TLA+] Verifying...")
                    res = verify_tla(current_blocks["tla"])
                    results["tla"] = res
                    if res.success:
                        print(f"✓ Passed")
                        self.best_blocks["tla"] = current_blocks["tla"]
                    else:
                        print(f"✗ Failed: {res.message}")
                        all_pass = False
                        failing_tools.append("tla")
            else:
                all_pass = False
                failing_tools.append("tla")

            # Lean
            if current_blocks.get("lean"):
                if current_blocks["lean"] == self.best_blocks["lean"]:
                    print("[Lean] Skipping re-verification (identical to last success).")
                    results["lean"] = VerificationResult(True, "Identical to last success")
                else:
                    print("\n[Lean] Verifying...")
                    res = verify_lean(current_blocks["lean"])
                    results["lean"] = res
                    if res.success:
                        print(f"✓ Passed")
                        self.best_blocks["lean"] = current_blocks["lean"]
                    else:
                        print(f"✗ Failed: {res.message}")
                        all_pass = False
                        failing_tools.append("lean")
            else:
                all_pass = False
                failing_tools.append("lean")

            # Z3 (Optional)
            if current_blocks.get("z3"):
                if current_blocks["z3"] == self.best_blocks["z3"]:
                    print("[Z3] Skipping re-verification (identical to last success).")
                    results["z3"] = VerificationResult(True, "Identical to last success")
                else:
                    print("\n[Z3] Verifying...")
                    res = verify_z3(current_blocks["z3"])
                    results["z3"] = res
                    if res.success:
                        print(f"✓ Passed")
                        self.best_blocks["z3"] = current_blocks["z3"]
                    else:
                        print(f"✗ Failed: {res.message}")
                        all_pass = False
                        failing_tools.append("z3")

            # Write detailed debug log
            with open(f"debug/iteration_{i+1}_results.txt", "w") as f:
                for tool, res in results.items():
                    f.write(f"=== {tool.upper()} ===\n")
                    f.write(str(res))
                    f.write("\n\n")

            # --- Success Handler ---
            if all_pass:
                # ... existing success logic ...
                print("\n[SUCCESS] All verifiers passed!")
                self.library.save_proof_set(task, current_blocks)
                
                # --- Final Analysis Step ---
                print("\nGenerating Logical Analysis...")
                analysis_prompt = (
                    "Great! The formal proofs have all passed verification. "
                    "Now, please summarize the answer to the user's original question. "
                    "Analyze the proofs: what specific invariants did we prove? "
                    "What assumptions does the argument rely on? "
                    "Under what conditions is this reasoning valid? "
                    "Present this as a clear, rigorous final answer."
                )
                final_analysis = self.proposer.propose(analysis_prompt, feedback=None) # sending prompt as task, no feedback
                
                print("\n========== FINAL LOGICAL ANALYSIS ==========\n")
                print(final_analysis)
                print("\n============================================")
                return True, final_analysis
            
            # --- Feedback ---
            feedback = f"The following verifiers failed: {', '.join([t.upper() for t in failing_tools])}.\n"
            for tool in failing_tools:
                res = results.get(tool)
                if res:
                    feedback += f"\n--- {tool.upper()} ERROR ---\n{res.message}\n{res.details}\n"
            
            feedback += f"\nPlease fix the issues in the failing components ({', '.join(failing_tools)}). You do NOT need to provide the other proofs if they have already passed."
            
            print("\n[RETRY] Sending feedback to LLM...")

        print("\n[FAILURE] Max iterations reached.")
        return False, None

import argparse
import sys
from src.verifiers.tla_verifier import verify_tla
from src.verifiers.lean_verifier import verify_lean
from src.verifiers.z3_verifier import verify_z3
from src.proposer.client import Proposer
from src.library_manager import LibraryManager
from src.verifiers.common import VerificationResult
import os

class FormalReasoningLoop:
    def __init__(self, max_iterations=5, backend="gemini", model=None, api_key=None, base_url=None, verbose=False):
        self.max_iterations = max_iterations
        self.verbose = verbose
        self.proposer = Proposer(backend=backend, model_name=model, api_key=api_key, base_url=base_url)
        self.library = LibraryManager()
        self.best_blocks = {"tla": None, "lean": None, "z3": None}
        if not os.path.exists("debug"):
            os.makedirs("debug")

    def run(self, task):
        feedback = None
        last_response = None
        for i in range(self.max_iterations):
            print(f"\n--- Iteration {i+1} ---")
            
            # Call LLM
            response = self.proposer.propose(task, feedback)
            last_response = response
            
            # Save raw response for debugging
            with open(f"debug/iteration_{i+1}_raw.txt", "w") as f:
                f.write(response)
            
            current_blocks = self.proposer.extract_code(response)
            
            # --- Stateful Memory & Completeness Check ---
            for kind in ["tla", "lean", "z3"]:
                if not current_blocks.get(kind) and self.best_blocks[kind]:
                    print(f"[{kind.upper()}] restoring previously successful block.")
                    current_blocks[kind] = self.best_blocks[kind]

            results = {}
            all_pass = True
            failing_tools = []
            
            # --- Verification ---

            # TLA+
            if current_blocks.get("tla"):
                if current_blocks["tla"] == self.best_blocks["tla"]:
                    print("[TLA+] Skipping re-verification (identical to last success).")
                    results["tla"] = VerificationResult(True, "Identical to last success")
                else:
                    print("\n[TLA+] Verifying...")
                    res = verify_tla(current_blocks["tla"])
                    results["tla"] = res
                    if res.success:
                        print(f"✓ Passed")
                        self.best_blocks["tla"] = current_blocks["tla"]
                    else:
                        print(f"✗ Failed: {res.message}")
                        if self.verbose:
                            print(res.details)
                        all_pass = False
                        failing_tools.append("tla")
            else:
                all_pass = False
                failing_tools.append("tla")

            # Lean
            if current_blocks.get("lean"):
                if current_blocks["lean"] == self.best_blocks["lean"]:
                    print("[Lean] Skipping re-verification (identical to last success).")
                    results["lean"] = VerificationResult(True, "Identical to last success")
                else:
                    print("\n[Lean] Verifying...")
                    res = verify_lean(current_blocks["lean"])
                    results["lean"] = res
                    if res.success:
                        print(f"✓ Passed")
                        self.best_blocks["lean"] = current_blocks["lean"]
                    else:
                        print(f"✗ Failed: {res.message}")
                        if self.verbose:
                            print(res.details)
                        all_pass = False
                        failing_tools.append("lean")
            else:
                all_pass = False
                failing_tools.append("lean")

            # Z3 (Optional)
            if current_blocks.get("z3"):
                if current_blocks["z3"] == self.best_blocks["z3"]:
                    print("[Z3] Skipping re-verification (identical to last success).")
                    results["z3"] = VerificationResult(True, "Identical to last success")
                else:
                    print("\n[Z3] Verifying...")
                    res = verify_z3(current_blocks["z3"])
                    results["z3"] = res
                    if res.success:
                        print(f"✓ Passed")
                        self.best_blocks["z3"] = current_blocks["z3"]
                    else:
                        print(f"✗ Failed: {res.message}")
                        if self.verbose:
                            print(res.details)
                        all_pass = False
                        failing_tools.append("z3")

            # Write detailed debug log
            with open(f"debug/iteration_{i+1}_results.txt", "w") as f:
                for tool, res in results.items():
                    f.write(f"=== {tool.upper()} ===\n")
                    f.write(str(res))
                    f.write("\n\n")

            # --- Success Handler ---
            if all_pass:
                print("\n[SUCCESS] All verifiers passed!")
                self.library.save_proof_set(task, current_blocks)
                
                # --- Final Analysis Step ---
                print("\nGenerating Logical Analysis...")
                analysis_prompt = (
                    "Great! The formal proofs have all passed verification. "
                    "Now, please summarize the answer to the user's original question. "
                    "Analyze the proofs: what specific invariants did we prove? "
                    "What assumptions does the argument rely on? "
                    "Under what conditions is this reasoning valid? "
                    "Present this as a clear, rigorous final answer."
                )
                final_analysis = self.proposer.propose(analysis_prompt, feedback=None)
                
                print("\n========== FINAL LOGICAL ANALYSIS ==========\n")
                print(final_analysis)
                print("\n============================================")
                return True, final_analysis
            
            # --- Feedback ---
            feedback = f"The following verifiers failed: {', '.join([t.upper() for t in failing_tools])}.\n"
            for tool in failing_tools:
                res = results.get(tool)
                if res:
                    feedback += f"\n--- {tool.upper()} ERROR ---\n{res.message}\n{res.details}\n"
            
            feedback += f"\nPlease fix the issues in the failing components ({', '.join(failing_tools)}). You do NOT need to provide the other proofs if they have already passed."
            
            print("\n[RETRY] Sending feedback to LLM...")

        print("\n[FAILURE] Max iterations reached.")
        print("\n========== LATEST ATTEMPT (UNVERIFIED) ==========\n")
        print(last_response)
        print("\n=================================================")
        return False, None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Formal Reasoning Loop")
    parser.add_argument("task", nargs="?", default="How could AI agents use formal methods to produce superhuman thinking?", help="The natural language query/task")
    parser.add_argument("--backend", default="gemini", choices=["gemini", "openai", "ollama"], help="LLM backend to use")
    parser.add_argument("--model", help="Specific model name (e.g., gpt-4, gemini-2.5-flash, llama3)")
    parser.add_argument("--base-url", help="Base URL for OpenAI/Ollama API")
    parser.add_argument("--api-key", help="API Key (overrides env vars)")
    parser.add_argument("--verbose", action="store_true", help="Show detailed verification errors in output")
    
    args = parser.parse_args()
    
    frl = FormalReasoningLoop(
        backend=args.backend,
        model=args.model,
        api_key=args.api_key,
        base_url=args.base_url,
        verbose=args.verbose
    )
    frl.run(args.task)