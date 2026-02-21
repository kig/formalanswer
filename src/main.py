import argparse
import sys
import os
from src.verifiers.tla_verifier import verify_tla
from src.verifiers.lean_verifier import verify_lean
from src.verifiers.python_verifier import verify_python
from src.verifiers.consistency_checker import check_consistency, check_structure
from src.verifiers.auto_repair import try_auto_repair
from src.proposer.client import Proposer
from src.proposer.retriever import Retriever
from src.library_manager import LibraryManager
from src.verifiers.common import VerificationResult

class FormalReasoningLoop:
    def __init__(self, max_iterations=5, backend="gemini", model=None, api_key=None, base_url=None, verbose=False, combat=False, peer_review=False, rap_battle=False, generate_rap=False, force_mode=None):
        self.max_iterations = max_iterations
        self.verbose = verbose
        self.combat = combat
        self.peer_review = peer_review
        self.rap_battle = rap_battle
        self.generate_rap = generate_rap
        self.force_mode = force_mode
        self.proposer = Proposer(backend=backend, model_name=model, api_key=api_key, base_url=base_url)
        self.retriever = Retriever()
        self.library = LibraryManager()
        self.best_blocks = {"tla": None, "lean": None, "python": None}
        if not os.path.exists("debug"):
            os.makedirs("debug")

    def finalize_rap_battle(self, task_dir):
        print("\n[RAP BATTLE] Constructing Final Track...")
        raw_history = ""
        raw_dir = os.path.join(task_dir, "raw")
        if os.path.exists(raw_dir):
            try:
                # Sort numerically 1.txt, 2.txt etc. ignoring potential labels like 1_label.txt for now or handling them
                # Split by first underscore or dot to get number
                files = sorted([f for f in os.listdir(raw_dir) if f.endswith('.txt')], key=lambda x: int(x.split('_')[0].split('.')[0]))
                for f in files:
                     with open(os.path.join(raw_dir, f), 'r') as rf:
                         raw_history += f"--- TURN {f} ---\n" + rf.read() + "\n\n"
            except Exception as e:
                print(f"Error reading raw history: {e}")
                return

        final_track = self.proposer.construct_final_rap(raw_history)
        print("\n========== FINAL RAP BATTLE TRACK ==========\n")
        print(final_track)
        print("\n============================================")
        with open(os.path.join(task_dir, "final_rap_battle.txt"), "w") as f:
            f.write(final_track)

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
            response = self.proposer.propose(task, feedback, context=context, rap_battle=self.rap_battle, combat=self.combat, peer_review=self.peer_review, force_mode=self.force_mode)
            last_response = response
            
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
            
            import re
            # --- Verification Helper ---
            def verify_blocks(kind, blocks, verifier_func):
                if not blocks:
                    return True 
                
                # Check if identical to best
                if blocks == self.best_blocks[kind]:
                    print(f"[{kind.upper()}] Skipping re-verification (identical to last success).")
                    results[kind] = [VerificationResult(True, "Identical to last success")] * len(blocks)
                    return True

                print(f"\n[{kind.upper()}] Verifying {len(blocks)} block(s)...")
                kind_results = []
                kind_pass = True
                
                for idx, block in enumerate(blocks):
                    # Check for empty/comment-only blocks
                    # Strip C-style, Python-style, and TLA-style comments roughly
                    clean_block = block
                    clean_block = re.sub(r'//.*', '', clean_block) # C++ style
                    clean_block = re.sub(r'--.*', '', clean_block) # SQL/Haskell/Lean/Lua style
                    clean_block = re.sub(r'#.*', '', clean_block)  # Python style
                    clean_block = re.sub(r'/\*.*?\*/', '', clean_block, flags=re.DOTALL) # Multi-line
                    clean_block = re.sub(r'\(\*.*?\*\)', '', clean_block, flags=re.DOTALL) # TLA/Pascal/ML style
                    
                    if not clean_block.strip():
                        print(f"  Block {idx+1}: ✓ Passed (Empty/Comment-only omitted)")
                        kind_results.append(VerificationResult(True, "Empty/Comment-only block treated as Pass"))
                        continue

                    res = verifier_func(block)
                    
                    # --- AUTO-REPAIR (Lean Only) ---
                    if not res.success and kind == "lean":
                        repair_success, fixed_code, repaired_res = try_auto_repair(block, res.details)
                        if repair_success:
                            print(f"  Block {idx+1}: 🔧 Auto-Repaired!")
                            # Update the block in the list so it gets saved correctly later if needed
                            blocks[idx] = fixed_code 
                            res = repaired_res
                    
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
                print("[TLA+] No block found. Skipping (assuming Factual/Historical omission).")

            # Lean
            if current_blocks.get("lean"):
                if verify_blocks("lean", current_blocks["lean"], verify_lean):
                    self.best_blocks["lean"] = current_blocks["lean"]
                else:
                    all_pass = False
                    failing_tools.append("lean")
            else:
                print("[LEAN] No block found. Skipping (assuming Factual/Historical omission).")

            # Python
            if current_blocks.get("python"):
                if verify_blocks("python", current_blocks["python"], verify_python):
                    self.best_blocks["python"] = current_blocks["python"]
                else:
                    all_pass = False
                    failing_tools.append("python")
            # Python is technically optional in prompt (Z3/Python), but if present must pass.
            if not current_blocks.get("python") and "python" in failing_tools:
                 pass # Already handled

            # Write detailed debug log
            with open(f"debug/iteration_{i+1}_results.txt", "w") as f:
                for kind, res_list in results.items():
                    f.write(f"=== {kind.upper()} ===\n")
                    for idx, res in enumerate(res_list):
                        f.write(f"Block {idx+1}:\n{str(res)}\n\n")

            # --- CONSISTENCY CHECK ---
            consistency_warnings = []
            if current_blocks.get("tla") and current_blocks.get("python"):
                 tla_full = "\n".join(current_blocks["tla"])
                 py_full = "\n".join(current_blocks["python"])
                 consistency_warnings = check_consistency(tla_full, py_full)
                 consistency_warnings.extend(check_structure(tla_full, py_full))
                 if consistency_warnings:
                     print(f"[CONSISTENCY] Found {len(consistency_warnings)} warnings.")
                     for w in consistency_warnings:
                         print(f"  {w}")

            # --- PREPARE FEEDBACK & LOGGING ---
            feedback_parts = []
            verification_errors = []
            combat_log = ""
            
            # 1. Verification Errors
            if not all_pass:
                verification_errors.append(f"The following verifiers failed: {', '.join([t.upper() for t in failing_tools])}.")
                for tool in failing_tools:
                    res_list = results.get(tool, [])
                    for idx, res in enumerate(res_list):
                        if not res.success:
                            verification_errors.append(f"\n--- {tool.upper()} BLOCK {idx+1} ERROR ---\n{res.message}\n{res.details}\n")
                verification_errors.append(f"Please fix the issues in the failing components ({', '.join(failing_tools)}).")

            # --- COMBAT MODE ---
            if self.combat and current_blocks.get("prose"):
                print("\n[COMBAT MODE] Initiating Adversarial Review...")
                
                proof_text = current_blocks.get("prose", "")
                
                # 1. The Red Team Attack
                print("  Red Team is analyzing...")
                objection = self.proposer.critique(proof_text)
                print(f"  Objection: {objection[:100]}...")
                
                # 2. The Judge's Verdict
                print("  Judge is deliberating...")
                score, commentary = self.proposer.judge(proof_text, objection)
                print(f"  Score: {score}/1.0")
                print(f"  Commentary: {commentary}")
                
                # Format log
                combat_log = f"\n\n=== COMBAT MODE REPORT ===\nOBJECTION:\n{objection}\n\nJUDGE COMMENTARY:\n{commentary}\n\nSCORE: {score}/1.0\n"
                
                if score < 0.7:
                    print("  [COMBAT RESULT] Argument Destroyed.")
                    all_pass = False 
                    combat_log += "RESULT: FAILED (Feedback Triggered)\n"
                    combat_feedback = f"REVIEWER OBJECTION:\n{objection}\n\nJUDGE COMMENTARY:\n{commentary}\n\nYour reasoning was found to be weak (Score: {score}). Please address this objection."
                    feedback_parts.append(combat_feedback)
                else:
                    print("  [COMBAT RESULT] Argument Survived.")
                    combat_log += "RESULT: PASSED\n"

            # --- PEER REVIEW MODE ---
            peer_feedback_content = None
            if self.peer_review and current_blocks.get("prose"):
                print("\n[PEER REVIEW MODE] Initiating Constructive Review...")
                
                proof_text = current_blocks.get("prose", "")
                
                # 1. The Peer Review
                print("  Peer Reviewer is analyzing...")
                suggestion = self.proposer.peer_review(proof_text)
                print(f"  Suggestion: {suggestion[:100]}...")
                
                # 2. The Editor's Verdict
                print("  Editor is assessing...")
                score = self.proposer.peer_review_judge(proof_text, suggestion)
                print(f"  Score: {score}/1.0")
                
                # Format log
                peer_log = f"\n\n=== PEER REVIEW REPORT ===\nSUGGESTION:\n{suggestion}\n\nSCORE: {score}/1.0\n"
                
                combat_log += peer_log
                peer_feedback_content = f"PEER REVIEW SUGGESTION:\n{suggestion}\n\nReviewer Score: {score}/1.0."

                if score < 0.8:
                    print("  [PEER REVIEW RESULT] Improvement Required.")
                    all_pass = False 
                    combat_log += "RESULT: FAILED (Feedback Triggered)\n"
                    feedback_parts.append(f"{peer_feedback_content}\nPLEASE ADDRESS THIS ISSUE.")
                else:
                    print("  [PEER REVIEW RESULT] Argument Robust.")
                    combat_log += "RESULT: PASSED\n"

            # --- RAP BATTLE MODE ---
            rap_roast_content = None
            if self.rap_battle and current_blocks.get("prose"):
                print("\n[RAP BATTLE MODE] Initiating Lyrical Combat...")
                
                proof_text = current_blocks.get("prose", "")
                
                # 1. The Roast
                print("  Opponent is spitting bars...")
                roast = self.proposer.rap_battle(proof_text)
                print(f"  Roast: {roast[:100]}...")
                
                # 2. The Judge's Score
                print("  Judge is reacting...")
                score, commentary = self.proposer.rap_judge(proof_text, roast)
                print(f"  Score: {score}/1.0")
                print(f"  Commentary: {commentary}")
                
                # Format log
                rap_log = f"\n\n=== RAP BATTLE REPORT ===\nROAST:\n{roast}\n\nJUDGE COMMENTARY:\n{commentary}\n\nSCORE: {score}/1.0\n"
                
                combat_log += rap_log
                rap_roast_content = f"RAP BATTLE OPPONENT:\n{roast}\n\nJUDGE COMMENTARY:\n{commentary}\n\nJudge Score: {score}/1.0."

                if score < 0.6:
                    print("  [RAP BATTLE RESULT] You got served.")
                    all_pass = False 
                    combat_log += "RESULT: FAILED (Feedback Triggered)\n"
                    feedback_parts.append(f"{rap_roast_content}\nFIX THE LOGIC OR GET ROASTED AGAIN.")
                else:
                    print("  [RAP BATTLE RESULT] You held your own.")
                    combat_log += "RESULT: PASSED\n"

            # Join feedback
            if verification_errors:
                feedback_parts.extend(verification_errors)
            
            if consistency_warnings:
                feedback_parts.append("\n[CONSISTENCY WARNINGS] (Fix mismatch between TLA+ and Python):\n" + "\n".join(consistency_warnings))

            if feedback_parts:
                if peer_feedback_content and not any("PEER REVIEW SUGGESTION" in part for part in feedback_parts):
                     feedback_parts.append(f"Additional Peer Feedback:\n{peer_feedback_content}")
                if rap_roast_content and not any("RAP BATTLE OPPONENT" in part for part in feedback_parts):
                     feedback_parts.append(f"Additional Rap Battle Roast:\n{rap_roast_content}")
                feedback = "\n\n".join(feedback_parts)
            else:
                feedback = None

            # Save raw response WITH combat log
            full_log = response + combat_log
            self.library.save_raw_response(task_dir, i + 1, full_log)

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
                    "3. **Methodology:** Briefly explain the modeling strategy (e.g., 'Modeled as a probabilistic state machine...').\n\n"
                    
                    "Do NOT repeat the 'Critique' or 'Rationale' sections from the previous step. "
                    "Focus on synthesizing the *verified truths* into a coherent narrative."
                    f"{python_output}"
                )
                final_analysis = self.proposer.propose(analysis_prompt, feedback=None, context=context, force_mode=self.force_mode)
                
                self.library.save_raw_response(task_dir, i + 1, final_analysis, label="final_analysis")
                
                if current_blocks.get("prose"):
                    current_blocks["prose"] = f"=== VERIFIED FORMAL ANSWER ===\n\n{final_analysis}"
                else:
                    current_blocks["prose"] = final_analysis

                self.library.save_proofs(task_dir, current_blocks, original_prompt=task)
                
                print("\n========== FINAL LOGICAL ANALYSIS ==========\n")
                print(final_analysis)
                print("\n============================================")
                
                if self.rap_battle or self.generate_rap:
                    self.finalize_rap_battle(task_dir)
                
                return True, final_analysis
            
            # Retry
            print("\n[RETRY] Sending feedback to LLM...")

        print("\n[FAILURE] Max iterations reached.")
        print("\n========== LATEST ATTEMPT (UNVERIFIED) ==========\n")
        print(last_response)
        print("\n=================================================")
        
        if self.rap_battle or self.generate_rap:
            self.finalize_rap_battle(task_dir)
        
        return False, None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Formal Reasoning Loop")
    parser.add_argument("task", nargs="*", help="The natural language query/task (can be multiple strings)")
    parser.add_argument("--prompt-file", help="Load the prompt from a file.")
    parser.add_argument("--backend", default="gemini", choices=["gemini", "openai", "ollama"], help="LLM backend to use")
    parser.add_argument("--model", help="Specific model name (e.g., gpt-4, gemini-2.5-flash, llama3)")
    parser.add_argument("--base-url", help="Base URL for OpenAI/Ollama API")
    parser.add_argument("--api-key", help="API Key (overrides env vars)")
    parser.add_argument("--verbose", action="store_true", help="Show detailed verification errors in output")
    parser.add_argument("--combat", action="store_true", help="Enable Adversarial Combat Mode (Red Team review)")
    parser.add_argument("--peer-review", action="store_true", help="Enable Constructive Peer Review Mode")
    parser.add_argument("--rap-battle", action="store_true", help="Enable Logic Rap Battle Mode")
    parser.add_argument("--mode", choices=["discrete", "probabilistic", "hybrid", "factual"], help="Force a specific reasoning mode")
    parser.add_argument("--max-iterations", type=int, default=5, help="Maximum number of reasoning iterations")
    parser.add_argument("--construct-rap", nargs='?', const='CURRENT', help="Construct rap lyrics from history. Provide directory path for existing task, or flag for current session.")
    
    args = parser.parse_args()
    
    if args.construct_rap and args.construct_rap != 'CURRENT':
        # Standalone mode: Generate rap from existing directory
        frl = FormalReasoningLoop(
            backend=args.backend,
            model=args.model,
            api_key=args.api_key,
            base_url=args.base_url
        )
        frl.finalize_rap_battle(args.construct_rap)
        sys.exit(0)

    task_parts = []
    if args.prompt_file:
        with open(args.prompt_file, 'r') as f:
            task_parts.append(f.read())
    if args.task:
        task_parts.append(" ".join(args.task))
        
    if not task_parts:
        task = "How could AI agents use formal methods to produce superhuman thinking?"
    else:
        task = "\n\n".join(task_parts)

    frl = FormalReasoningLoop(
        max_iterations=args.max_iterations,
        backend=args.backend,
        model=args.model,
        api_key=args.api_key,
        base_url=args.base_url,
        verbose=args.verbose,
        combat=args.combat,
        peer_review=args.peer_review,
        rap_battle=args.rap_battle,
        generate_rap=(args.construct_rap == 'CURRENT'),
        force_mode=args.mode
    )
    frl.run(task)
