import concurrent.futures
import re
from typing import List, Callable, Dict, Any
from src.verifiers.common import VerificationResult
from src.verifiers.auto_repair import try_auto_repair
from src.utils.logger import console, log_info, log_success, log_error

class VerifierRunner:
    def __init__(self, max_workers=4):
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)

    def verify_parallel(self, kind: str, blocks: List[str], verifier_func: Callable) -> List[VerificationResult]:
        if not blocks:
            return []
            
        log_info(f"[{kind.upper()}] Verifying {len(blocks)} block(s) in parallel...")
        
        futures = {}
        results = [None] * len(blocks)
        
        for idx, block in enumerate(blocks):
            # Check for empty blocks (simple heuristic to skip)
            clean_block = re.sub(r'//.*', '', block)
            clean_block = re.sub(r'--.*', '', clean_block)
            clean_block = re.sub(r'#.*', '', clean_block)
            if not clean_block.strip():
                results[idx] = VerificationResult(True, "Empty/Comment-only block treated as Pass")
                log_success(f"  Block {idx+1}: ✓ Passed (Empty)")
                continue

            # Unique ID for temp files to ensure thread safety
            # E.g., temp_lean_0.lean
            uid = f"temp_{kind}_{idx}"
            
            # Submit task
            if kind == "tla":
                # TLA verifier takes module_name (no extension)
                future = self.executor.submit(verifier_func, block, module_name=uid)
            else:
                # Lean or Python verifiers take filename (with extension)
                ext = "lean" if kind == "lean" else "py"
                filename = f"{uid}.{ext}"
                future = self.executor.submit(verifier_func, block, filename=filename)
                
            futures[future] = (idx, uid)
            
        # Process results as they complete
        for future in concurrent.futures.as_completed(futures):
            idx, uid = futures[future]
            try:
                res = future.result()
                
                # Auto-Repair for Lean
                if not res.success and kind == "lean":
                    # Synchronous retry to avoid complexity (blocking the main thread is fine here)
                    # We reuse the same filename
                    repair_filename = f"{uid}.lean"
                    success, fixed_code, repaired_res = try_auto_repair(blocks[idx], res.details, filename=repair_filename)
                    if success:
                        log_success(f"  Block {idx+1}: 🔧 Auto-Repaired!")
                        blocks[idx] = fixed_code # Update the block content in place
                        res = repaired_res
                
                results[idx] = res
                
                # Immediate feedback
                if res.success:
                    log_success(f"  Block {idx+1}: ✓ Passed")
                    if kind == "python" and res.details:
                         console.print(f"  --- Output ---\n{res.details}\n  --------------", style="dim")
                else:
                    log_error(f"  Block {idx+1}: ✗ Failed: {res.message}")
                     
            except Exception as e:
                results[idx] = VerificationResult(False, f"Exception during verification: {e}")
                log_error(f"  Block {idx+1}: ✗ Exception: {e}")
                
        return results
        
    def shutdown(self):
        self.executor.shutdown(wait=True)
