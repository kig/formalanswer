import argparse
import sys
import os
from src.verifiers.tla_verifier import verify_tla
from src.verifiers.lean_verifier import verify_lean
from src.verifiers.python_verifier import verify_python
from src.proposer.client import Proposer
from src.proposer.retriever import Retriever
from src.library_manager import LibraryManager
from src.verifiers.common import VerificationResult

class FormalReasoningLoop:
    def __init__(self, max_iterations=5, backend="gemini", model=None, api_key=None, base_url=None, verbose=False):
        self.max_iterations = max_iterations
        self.verbose = verbose
        self.proposer = Proposer(backend=backend, model_name=model, api_key=api_key, base_url=base_url)
        self.retriever = Retriever()
        self.library = LibraryManager()
        self.best_blocks = {"tla": None, "lean": None, "python": None}
        if not os.path.exists("debug"):
            os.makedirs("debug")

    def run(self, task):
        feedback = None
        last_response = None
        
        # Initialize session directory
        task_dir = self.library.init_task_dir(task)
        print(f"[SESSION] Initialized task directory: {task_dir}")
        
        # --- Context Retrieval ---
        print("\n[RETRIEVER] Searching for relevant formal modules...")
        context = self.retriever.retrieve(task)
        if context:
            print("[RETRIEVER] Found relevant modules. Injecting context.")
            # Optionally log context
            with open(os.path.join(task_dir, "context.txt"), "w") as f:
                f.write(context)
        else:
            print("[RETRIEVER] No existing modules found. Starting from scratch.")
        
        for i in range(self.max_iterations):
            print(f"\n--- Iteration {i+1} ---")
            
            # Call LLM with Context
            response = self.proposer.propose(task, feedback, context=context)
            last_response = response
            
            # Save raw response immediately
            self.library.save_raw_response(task_dir, i + 1, response)
            
            # Keep existing debug logging for backward compatibility
            with open(f"debug/iteration_{i+1}_raw.txt", "w") as f:
                f.write(response)
            
            current_blocks = self.proposer.extract_code(response)
            
            # --- Stateful Memory & Completeness Check ---
            for kind in ["tla", "lean", "python"]:
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

            # Python/Z3 (Empirical Grounding)
            if current_blocks.get("python"):
                if current_blocks["python"] == self.best_blocks["python"]:
                    print("[PYTHON] Skipping re-verification (identical to last success).")
                    results["python"] = VerificationResult(True, "Identical to last success")
                else:
                    print("\n[PYTHON] Verifying...")
                    res = verify_python(current_blocks["python"])
                    results["python"] = res
                    if res.success:
                        print(f"✓ Passed")
                        self.best_blocks["python"] = current_blocks["python"]
                    else:
                        print(f"✗ Failed: {res.message}")
                        if self.verbose:
                            print(res.details)
                        all_pass = False
                        failing_tools.append("python")

            # Write detailed debug log
            with open(f"debug/iteration_{i+1}_results.txt", "w") as f:
                for tool, res in results.items():
                    f.write(f"=== {tool.upper()} ===\n")
                    f.write(str(res))
                    f.write("\n\n")

            # --- Success Handler ---
            if all_pass:
                print("\n[SUCCESS] All verifiers passed!")
                
                # --- Final Analysis Step ---
                print("\nGenerating Verified Prose Answer...")
                analysis_prompt = (
                    "The formal proofs (TLA+, Lean, Z3) have all passed verification. "
                    "Construct the FINAL ANSWER to the user's original question based EXCLUSIVELY on these verified results.\n\n"
                    "Your answer must:\n"
                    "1. Directly answer the original question.\n"
                    "2. Explain the 'Shared Invariant' that was formally proven.\n"
                    "3. Cite the specific results from the TLA+ temporal safety check and the Lean 4 arithmetic proof.\n"
                    "4. State the boundary conditions (the 'Shared Constants') under which this answer is mathematically guaranteed to be true.\n\n"
                    "Do not make any claims that were not covered by the formal verification. Present this as a rigorous, superhumanly precise 'Verified Answer'."
                )
                final_analysis = self.proposer.propose(analysis_prompt, feedback=None, context=context)
                
                # Save final analysis raw response
                self.library.save_raw_response(task_dir, i + 1, final_analysis, label="final_analysis")
                
                # Combine original prose and final analysis for the library
                if current_blocks.get("prose"):
                    current_blocks["prose"] = f"{current_blocks['prose']}\n\n=== FINAL LOGICAL ANALYSIS ===\n\n{final_analysis}"
                else:
                    current_blocks["prose"] = final_analysis

                self.library.save_proofs(task_dir, current_blocks)
                
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
    parser.add_argument("--prompt-file", help="Load the prompt from a file.")
    parser.add_argument("--backend", default="gemini", choices=["gemini", "openai", "ollama"], help="LLM backend to use")
    parser.add_argument("--model", help="Specific model name (e.g., gpt-4, gemini-2.5-flash, llama3)")
    parser.add_argument("--base-url", help="Base URL for OpenAI/Ollama API")
    parser.add_argument("--api-key", help="API Key (overrides env vars)")
    parser.add_argument("--verbose", action="store_true", help="Show detailed verification errors in output")
    
    args = parser.parse_args()
    task = args.task

    if args.prompt_file:
        with open(args.prompt_file, 'r') as f:
            task = f.read()

    frl = FormalReasoningLoop(
        backend=args.backend,
        model=args.model,
        api_key=args.api_key,
        base_url=args.base_url,
        verbose=args.verbose
    )
    frl.run(task)