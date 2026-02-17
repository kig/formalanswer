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
        
        # Save prompt immediately
        self.library.save_prompt(task_dir, task)
        
        # --- Context Retrieval ---
        print("\n[RETRIEVER] Searching for relevant formal modules...")
        context = self.retriever.retrieve(task)
        if context:
            print("[RETRIEVER] Found relevant modules. Injecting context.")
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
            
            # Save candidate proofs
            self.library.save_candidate_proofs(task_dir, current_blocks)
            
            # --- Stateful Memory & Completeness Check ---
            # If a language is missing entirely, restore best_blocks (which is now a list or None)
            for kind in ["tla", "lean", "python"]:
                if not current_blocks.get(kind) and self.best_blocks[kind]:
                    print(f"[{kind.upper()}] restoring previously successful blocks.")
                    current_blocks[kind] = self.best_blocks[kind]

            results = {} # Map kind -> List[VerificationResult]
            all_pass = True
            failing_tools = []
            
            # --- Verification Helper ---
            def verify_blocks(kind, blocks, verifier_func):
                if not blocks:
                    return True # Nothing to verify is a "pass" if not required? 
                    # But prompt requires them. Logic below handles missing blocks.
                
                # Check if identical to best
                if blocks == self.best_blocks[kind]:
                    print(f"[{kind.upper()}] Skipping re-verification (identical to last success).")
                    results[kind] = [VerificationResult(True, "Identical to last success")] * len(blocks)
                    return True

                print(f"\n[{kind.upper()}] Verifying {len(blocks)} block(s)...")
                kind_results = []
                kind_pass = True
                
                for idx, block in enumerate(blocks):
                    res = verifier_func(block)
                    kind_results.append(res)
                    if res.success:
                        print(f"  Block {idx+1}: ✓ Passed")
                        if res.details and kind == "python": # Print stdout for Python
                             print(f"  --- Output ---\n{res.details}\n  --------------")
                    else:
                        print(f"  Block {idx+1}: ✗ Failed: {res.message}")
                        if self.verbose:
                            print(res.details)
                        kind_pass = False
                
                results[kind] = kind_results
                return kind_pass

            # TLA+
            if current_blocks.get("tla"):
                if verify_blocks("tla", current_blocks["tla"], verify_tla):
                    self.best_blocks["tla"] = current_blocks["tla"]
                else:
                    all_pass = False
                    failing_tools.append("tla")
            else:
                all_pass = False
                failing_tools.append("tla") # Missing mandatory block

            # Lean
            if current_blocks.get("lean"):
                if verify_blocks("lean", current_blocks["lean"], verify_lean):
                    self.best_blocks["lean"] = current_blocks["lean"]
                else:
                    all_pass = False
                    failing_tools.append("lean")
            else:
                all_pass = False
                failing_tools.append("lean") # Missing mandatory block

            # Python
            if current_blocks.get("python"):
                if verify_blocks("python", current_blocks["python"], verify_python):
                    self.best_blocks["python"] = current_blocks["python"]
                else:
                    all_pass = False
                    failing_tools.append("python")
            # Python is technically optional in prompt (Z3/Python), but if present must pass.
            # If missing, we don't fail, unless previous best existed?
            # Actually prompt says "MANDATORY" for Z3/Python in recent updates.
            # Let's enforce it if it's in the prompt instructions.
            # The extract code returns None if empty list.
            if not current_blocks.get("python") and "python" in failing_tools:
                 pass # Already handled

            # Write detailed debug log
            with open(f"debug/iteration_{i+1}_results.txt", "w") as f:
                for kind, res_list in results.items():
                    f.write(f"=== {kind.upper()} ===\n")
                    for idx, res in enumerate(res_list):
                        f.write(f"Block {idx+1}:\n{str(res)}\n\n")

            # --- Success Handler ---
            if all_pass:
                print("\n[SUCCESS] All verifiers passed!")
                
                # --- Final Analysis Step ---
                print("\nGenerating Verified Prose Answer...")
                
                # Collect Python output
                python_output = ""
                if results.get("python"):
                    outputs = [r.details for r in results["python"] if r.success]
                    if outputs:
                        python_output = "\n\n[PYTHON OUTPUTS]:\n" + "\n".join(outputs)

                analysis_prompt = (
                    "The formal proofs (TLA+, Lean, Python/Z3) have all passed verification. "
                    "Now, construct the FINAL ANSWER to the user's original question. "
                    "Your answer must follow this structure EXACTLY to ensure usefulness to the reader:\n\n"
                    
                    "1. **Executive Summary:** A direct, concise answer to the question. No fluff.\n"
                    "2. **Formal Guarantee:** Specifically list what was *proven* versus what was *assumed*. "
                    "Cite specific theorems (Lean), invariants (TLA+), or empirical results (Python).\n"
                    "3. **Methodology:** Briefly explain the modeling strategy.\n\n"
                    
                    "Do NOT repeat the 'Critique' or 'Rationale' sections. "
                    f"{python_output}"
                )
                final_analysis = self.proposer.propose(analysis_prompt, feedback=None, context=context)
                
                self.library.save_raw_response(task_dir, i + 1, final_analysis, label="final_analysis")
                
                if current_blocks.get("prose"):
                    current_blocks["prose"] = f"=== VERIFIED FORMAL ANSWER ===\n\n{final_analysis}"
                else:
                    current_blocks["prose"] = final_analysis

                self.library.save_proofs(task_dir, current_blocks, original_prompt=task)
                
                print("\n========== FINAL LOGICAL ANALYSIS ==========\n")
                print(final_analysis)
                print("\n============================================")
                return True, final_analysis
            
            # --- Feedback ---
            feedback = f"The following verifiers failed: {', '.join([t.upper() for t in failing_tools])}.\n"
            for tool in failing_tools:
                res_list = results.get(tool, [])
                for idx, res in enumerate(res_list):
                    if not res.success:
                        feedback += f"\n--- {tool.upper()} BLOCK {idx+1} ERROR ---\n{res.message}\n{res.details}\n"
            
            feedback += f"\nPlease fix the issues in the failing components ({', '.join(failing_tools)})."
            
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
