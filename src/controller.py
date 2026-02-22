import os
import re
from src.verifiers.tla_verifier import verify_tla
from src.verifiers.lean_verifier import verify_lean
from src.verifiers.python_verifier import verify_python
from src.verifiers.consistency_checker import check_consistency, check_structure
from src.runner import VerifierRunner
from src.proposer.client import Proposer
from src.proposer.retriever import Retriever
from src.library_manager import LibraryManager
from src.verifiers.common import VerificationResult
from src.utils.logger import log_info, log_success, log_error, log_warning, log_section, status, console
from src.utils.meter import Meter
from src.ui.reporter import Reporter

class FormalReasoningLoop:
    def __init__(self, max_iterations=None, backend="gemini", model=None, api_key=None, base_url=None, verbose=False, show_prompts=False, combat=None, peer_review=None, rap_battle=False, generate_rap=False, force_mode=None, tier="pro"):
        self.tier = tier.lower()
        
        # Apply Tier Presets
        if self.tier == "standard":
            self.max_iterations = max_iterations if max_iterations is not None else 1
            self.combat = combat if combat is not None else False
            self.peer_review = peer_review if peer_review is not None else False
        elif self.tier == "enterprise":
            self.max_iterations = max_iterations if max_iterations is not None else 10
            self.combat = combat if combat is not None else True
            self.peer_review = peer_review if peer_review is not None else True
        else: # Pro (Default)
            self.max_iterations = max_iterations if max_iterations is not None else 5
            self.combat = combat if combat is not None else True
            self.peer_review = peer_review if peer_review is not None else False

        self.verbose = verbose
        self.show_prompts = show_prompts
        self.rap_battle = rap_battle
        self.generate_rap = generate_rap
        self.force_mode = force_mode
        self.proposer = Proposer(backend=backend, model_name=model, api_key=api_key, base_url=base_url)
        self.retriever = Retriever()
        self.runner = VerifierRunner()
        self.library = LibraryManager()
        self.meter = Meter()
        self.best_blocks = {"tla": None, "lean": None, "python": None}
        if not os.path.exists("debug"):
            os.makedirs("debug")

    def finalize_rap_battle(self, task_dir):
        log_info("[RAP BATTLE] Constructing Final Track...")
        raw_history = ""
        raw_dir = os.path.join(task_dir, "raw")
        if os.path.exists(raw_dir):
            try:
                # Sort numerically 1.txt, 2.txt etc. ignoring potential labels like 1_label.txt for now or handling them
                # Split by first underscore or dot to get number
                files = sorted([f for f in os.listdir(raw_dir) if f.endswith('.txt')], key=lambda x: int(x.split('_')[0].split('.')[0]))
                for f in files:
                     with open(os.path.join(raw_dir, f), 'r') as rf:
                         raw_history += f"--- TURN {f} ---\\n" + rf.read() + "\\n\\n"
            except Exception as e:
                log_error(f"Error reading raw history: {e}")
                return

        final_track = self.proposer.construct_final_rap(raw_history)
        log_section("FINAL RAP BATTLE TRACK", final_track, style="magenta")
        with open(os.path.join(task_dir, "final_rap_battle.txt"), "w") as f:
            f.write(final_track)
            
        # Tencent SongGeneration Format
        log_info("[RAP BATTLE] Formatting for Tencent SongGeneration...")
        formatted_track = self.proposer.produce_song_gen_lyrics(final_track)
        with open(os.path.join(task_dir, "final_rap_battle_formatted.txt"), "w") as f:
            f.write(formatted_track)
        log_success("Saved formatted lyrics to final_rap_battle_formatted.txt")

    def run(self, task):
        feedback = None
        last_response = None
        
        # Initialize session directory
        task_dir = self.library.init_task_dir(task)
        log_info(f"[SESSION] Initialized task directory: {task_dir}")
        
        # Save prompt immediately
        self.library.save_prompt(task_dir, task)
        
        # --- Context Retrieval ---
        log_info("[RETRIEVER] Searching for relevant formal modules...")
        context = self.retriever.retrieve(task)
        if context:
            log_success("[RETRIEVER] Found relevant modules. Injecting context.")
            with open(os.path.join(task_dir, "context.txt"), "w") as f:
                f.write(context)
        else:
            log_info("[RETRIEVER] No existing modules found. Starting from scratch.")
        
        for i in range(self.max_iterations):
            console.rule(f"Iteration {i+1}")
            
            # Call LLM with Context
            with status("Proposer is thinking..."):
                if self.show_prompts:
                    from src.proposer.prompts import format_user_prompt, RAP_BATTLE_RULES, ADVERSARIAL_COMBAT_RULES, PEER_REVIEW_RULES
                    from src.proposer.repair_prompt import REPAIR_PROMPT
                    from src.proposer.rap_repair_prompt import RAP_REPAIR_PROMPT
                    
                    context_prefix = ""
                    if self.rap_battle:
                        context_prefix = RAP_BATTLE_RULES
                    elif self.combat:
                        context_prefix = ADVERSARIAL_COMBAT_RULES
                    elif self.peer_review:
                        context_prefix = PEER_REVIEW_RULES

                    tier_prefix = ""
                    if self.tier == "standard":
                        tier_prefix = "ASSURANCE TIER: STANDARD. Focus on rapid, direct answers. Minimalist formalization. Be concise.\n"
                    elif self.tier == "enterprise":
                        tier_prefix = "ASSURANCE TIER: ENTERPRISE. Maximum rigor required. Exhaustive TLA+ specs and Lean proofs. Model every edge case. High-budget reasoning.\n"
                    else:
                        tier_prefix = "ASSURANCE TIER: PRO. Balanced rigor and cost. Robust formalization.\n"

                    if feedback:
                        repair_tmpl = RAP_REPAIR_PROMPT if self.rap_battle else REPAIR_PROMPT
                        display_prompt = f"{context_prefix}{tier_prefix}{repair_tmpl.format(feedback=feedback)}"
                    else:
                        display_prompt = format_user_prompt(task, context, force_mode=self.force_mode, context_prefix=context_prefix, tier_prefix=tier_prefix)
                    
                    log_section("OUTGOING PROMPT", display_prompt, style="cyan")

                response_obj = self.proposer.propose(task, feedback, context=context, rap_battle=self.rap_battle, combat=self.combat, peer_review=self.peer_review, force_mode=self.force_mode, tier=self.tier)
                response = response_obj.content
            
            if self.verbose:
                log_section("RAW LLM RESPONSE", response, style="dim")
                
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
                    log_info(f"[{kind.upper()}] restoring previously successful blocks.")
                    current_blocks[kind] = self.best_blocks[kind]

            results = {} # Map kind -> List[VerificationResult]
            all_pass = True
            failing_tools = []
            
            # --- Verification Helper ---
            def verify_blocks(kind, blocks, verifier_func):
                if not blocks:
                    return True 
                
                # Check if identical to best
                if blocks == self.best_blocks[kind]:
                    log_info(f"[{kind.upper()}] Skipping re-verification (identical to last success).")
                    results[kind] = [VerificationResult(True, "Identical to last success")] * len(blocks)
                    return True

                # Delegate to parallel runner
                kind_results = self.runner.verify_parallel(kind, blocks, verifier_func)
                
                results[kind] = kind_results
                return all(r.success for r in kind_results)

            # TLA+
            if current_blocks.get("tla"):
                if verify_blocks("tla", current_blocks["tla"], verify_tla):
                    self.best_blocks["tla"] = current_blocks["tla"]
                else:
                    all_pass = False
                    failing_tools.append("tla")
            else:
                log_info("[TLA+] No block found. Skipping (assuming Factual/Historical omission).")

            # Lean
            if current_blocks.get("lean"):
                if verify_blocks("lean", current_blocks["lean"], verify_lean):
                    self.best_blocks["lean"] = current_blocks["lean"]
                else:
                    all_pass = False
                    failing_tools.append("lean")
            else:
                log_info("[LEAN] No block found. Skipping (assuming Factual/Historical omission).")

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
                    f.write(f"=== {kind.upper()} ===\\n")
                    for idx, res in enumerate(res_list):
                        f.write(f"Block {idx+1}:\\n{str(res)}\\n\\n")

            # --- CONSISTENCY CHECK ---
            consistency_warnings = []
            if current_blocks.get("tla") and current_blocks.get("python"):
                 tla_full = "\\n".join(current_blocks["tla"])
                 py_full = "\\n".join(current_blocks["python"])
                 consistency_warnings = check_consistency(tla_full, py_full)
                 consistency_warnings.extend(check_structure(tla_full, py_full))
                 if consistency_warnings:
                     log_warning(f"[CONSISTENCY] Found {len(consistency_warnings)} warnings.")
                     for w in consistency_warnings:
                         log_warning(f"  {w}")

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
                            error_msg = f"\\n--- {tool.upper()} BLOCK {idx+1} ERROR ---\\n{res.message}\\n{res.details}\\n"
                            
                            # TLA+ Explain Trace
                            if tool == "tla" and "[COUNTER-EXAMPLE TRACE]:" in res.details:
                                try:
                                    match = re.search(r"\[COUNTER-EXAMPLE TRACE\]:\\n(.*)", res.details, re.DOTALL)
                                    if match:
                                        trace = match.group(1)
                                        # Safely access spec (handle if list length mismatch, though unlikely)
                                        if idx < len(current_blocks.get("tla", [])):
                                            spec = current_blocks["tla"][idx]
                                            log_info("  [TLA+] Explaining counter-example trace...")
                                            explanation = self.proposer.explain_trace(trace, spec)
                                            error_msg += f"\\n[TRACE EXPLANATION]:\\n{explanation}\\n"
                                except Exception as e:
                                    log_error(f"  [TLA+] Error explaining trace: {e}")

                            verification_errors.append(error_msg)
                verification_errors.append(f"Please fix the issues in the failing components ({', '.join(failing_tools)}).")

            # --- COMBAT MODE ---
            if self.combat and current_blocks.get("prose"):
                log_section("Combat Mode", "Initiating Adversarial Review...", style="red")
                
                proof_text = current_blocks.get("prose", "")
                
                # 1. The Red Team Attack
                with status("Red Team is analyzing..."):
                    objection = self.proposer.critique(proof_text)
                log_info(f"  Objection: {objection[:100]}...")
                
                # 2. The Judge's Verdict
                with status("Judge is deliberating..."):
                    score, commentary = self.proposer.judge(proof_text, objection)
                
                log_section("JUDGE VERDICT", f"**Score:** {score}/1.0\n\n**Commentary:**\n{commentary}", style="yellow")
                
                # Format log
                combat_log = f"\\n\\n=== COMBAT MODE REPORT ===\\nOBJECTION:\\n{objection}\\n\\nJUDGE COMMENTARY:\\n{commentary}\\n\\nSCORE: {score}/1.0\\n"
                
                if score < 0.7:
                    log_error("  [COMBAT RESULT] Argument Destroyed.")
                    all_pass = False 
                    combat_log += "RESULT: FAILED (Feedback Triggered)\\n"
                    combat_feedback = f"REVIEWER OBJECTION:\\n{objection}\\n\\nJUDGE COMMENTARY:\\n{commentary}\\n\\nYour reasoning was found to be weak (Score: {score}). Please address this objection."
                    feedback_parts.append(combat_feedback)
                else:
                    log_success("  [COMBAT RESULT] Argument Survived.")
                    combat_log += "RESULT: PASSED\\n"

            # Join feedback
            if verification_errors:
                feedback_parts.extend(verification_errors)
            
            if consistency_warnings:
                feedback_parts.append("\\n[CONSISTENCY WARNINGS] (Fix mismatch between TLA+ and Python):\\n" + "\\n".join(consistency_warnings))

            if feedback_parts:
                feedback = "\\n\\n".join(feedback_parts)
            else:
                feedback = None

            # Save raw response WITH combat log
            full_log = response + combat_log
            self.library.save_raw_response(task_dir, i + 1, full_log)

            # --- USAGE METERING ---
            usage = self.proposer.get_usage()
            self.meter.record_usage(self.proposer.model_name, usage.input_tokens, usage.output_tokens)
            self.proposer.usage_input = 0
            self.proposer.usage_output = 0
            log_info(f"[METER] {self.meter.get_session_summary()}")

            # --- Success Handler ---
            if all_pass:
                log_success("All verifiers passed!")
                
                # --- Final Analysis Step ---
                # Collect Python output
                python_output = ""
                if results.get("python"):
                    outputs = [r.details for r in results["python"] if r.success]
                    if outputs:
                        python_output = "\\n\\n[PYTHON OUTPUTS]:\\n" + "\\n".join(outputs)

                analysis_prompt = (
                    "The formal proofs (TLA+, Lean, Python/Z3) have all passed verification. "
                    "Now, construct the FINAL ANSWER to the user's original question. "
                    "Your answer must follow this structure EXACTLY to ensure usefulness to the reader:\\n\\n"
                    
                    "1. **Executive Summary:** A direct, concise answer to the question. No fluff.\\n"
                    "2. **Formal Guarantee:** Specifically list what was *proven* versus what was *assumed*. "
                    "Cite specific theorems (Lean), invariants (TLA+), or empirical results (Python).\\n"
                    "3. **Methodology:** Briefly explain the modeling strategy (e.g., 'Modeled as a probabilistic state machine...').\\n\\n"
                    
                    "Do NOT repeat the 'Critique' or 'Rationale' sections from the previous step. "
                    "Focus on synthesizing the *verified truths* into a coherent narrative."
                    f"{python_output}"
                )
                
                with status("Generating Verified Prose Answer..."):
                    resp_obj = self.proposer.propose(analysis_prompt, feedback=None, context=context, force_mode=self.force_mode, rap_battle=self.rap_battle, combat=self.combat, peer_review=self.peer_review, tier=self.tier)
                    final_analysis = resp_obj.content
                
                # Append formal proofs to final analysis for complete visibility
                formal_proofs_section = "\n\n# Verified Formal Proofs\n"
                if self.best_blocks.get("tla"):
                    formal_proofs_section += "\n## TLA+ Specification\n```tla\n" + "\n\n".join(self.best_blocks["tla"]) + "\n```\n"
                if self.best_blocks.get("lean"):
                    formal_proofs_section += "\n## Lean 4 Proof\n```lean\n" + "\n\n".join(self.best_blocks["lean"]) + "\n```\n"
                if self.best_blocks.get("python"):
                    formal_proofs_section += "\n## Python/Z3 Script\n```python\n" + "\n\n".join(self.best_blocks["python"]) + "\n```\n"
                
                final_analysis += formal_proofs_section

                self.library.save_raw_response(task_dir, i + 1, final_analysis, label="final_analysis")
                
                if current_blocks.get("prose"):
                    current_blocks["prose"] = f"=== VERIFIED FORMAL ANSWER ===\\n\\n{final_analysis}"
                else:
                    current_blocks["prose"] = final_analysis

                self.library.save_proofs(task_dir, current_blocks, original_prompt=task)
                
                log_section("FINAL LOGICAL ANALYSIS", final_analysis, style="green")
                
                # Final Usage Sync & Persist
                usage = self.proposer.get_usage()
                self.meter.record_usage(self.proposer.model_name, usage.input_tokens, usage.output_tokens)
                self.meter.persist(task)
                
                # Generate Professional Report
                report = Reporter.generate_report(task, final_analysis, results, self.meter.get_session_summary(), self.tier)
                with open(os.path.join(task_dir, "logic_report.md"), "w") as f:
                    f.write(report)
                
                log_success(f"Saved professional logic report to {task_dir}/logic_report.md")
                log_success(f"[METER] Task Complete. {self.meter.get_session_summary()}")

                if self.rap_battle or self.generate_rap:
                    self.finalize_rap_battle(task_dir)
                
                return True, final_analysis
            
            # Retry
            log_warning("Sending feedback to LLM...")

        log_error("Max iterations reached.")
        log_section("LATEST ATTEMPT (UNVERIFIED)", str(last_response), style="red")
        
        # Final Usage Persist even on failure
        usage = self.proposer.get_usage()
        self.meter.record_usage(self.proposer.model_name, usage.input_tokens, usage.output_tokens)
        self.meter.persist(task)
        
        # Generate Report for failure too
        report = Reporter.generate_report(task, str(last_response), results, self.meter.get_session_summary(), self.tier)
        with open(os.path.join(task_dir, "logic_report.md"), "w") as f:
            f.write(report)
            
        log_warning(f"[METER] Task Failed. {self.meter.get_session_summary()}")

        if self.rap_battle or self.generate_rap:
            self.finalize_rap_battle(task_dir)
        
        return False, None