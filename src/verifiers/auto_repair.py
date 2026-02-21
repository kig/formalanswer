import re
import shutil
import os
from typing import Tuple, Optional, List
from .lean_verifier import verify_lean
from .common import VerificationResult

def try_auto_repair(code_content: str, error_msg: str, filename: str = "temp.lean") -> Tuple[bool, str, Optional[VerificationResult]]:
    """
    Attempts to fix common Lean 4 tactic failures by substituting stronger tactics.
    Currently focuses on arithmetic and logical simplification failures.
    
    Args:
        code_content: The original Lean code.
        error_msg: The error message from the failed verification.
        filename: unique filename to use for verification to avoid collisions.
        
    Returns:
        (success, fixed_code, verification_result)
        If success is True, fixed_code contains the repaired source and verification_result contains the success details.
        If False, returns original code and None.
    """
    
    # 1. Parse the error to find the failing line
    # Standard Lean format: "filename:line:col: error: ..."
    # We look for the first error line
    line_match = re.search(r":(\d+):\d+:\s+error:", error_msg)
    if not line_match:
        return False, code_content, None
        
    line_idx = int(line_match.group(1)) - 1 # 0-indexed
    lines = code_content.split('\n')
    
    if line_idx < 0 or line_idx >= len(lines):
        return False, code_content, None
        
    failing_line = lines[line_idx]
    indentation = re.match(r"^\s*", failing_line).group(0)
    stripped_line = failing_line.strip()
    
    # 2. Define Tactics to Try (Hierarchy of Power)
    # If the line uses Key, try replacing it with Value
    # This is a simplisitic "whole line replacement" strategy for now
    
    replacements = []
    
    # If using 'linarith', try 'omega' (stronger for Nat/Int)
    if "linarith" in stripped_line:
        replacements.append("omega")
        replacements.append("zify at *; linarith")

    # If using 'ring', try 'linarith' or 'omega'
    elif "ring" in stripped_line:
        replacements.append("linarith")
        replacements.append("omega")
        
    # If using 'simp', try 'aesop' (search) or 'dsimp'
    elif "simp" in stripped_line:
        replacements.append("aesop")
        replacements.append("dsimp")
        
    # If just 'rfl' fails (often used for equality), try 'simp' or 'decide'
    elif stripped_line == "rfl":
        replacements.append("simp")
        replacements.append("decide")

    # 3. Apply Replacements
    for new_tactic in replacements:
        # Construct the new line preserving indentation
        # We assume the failing line is just the tactic call, e.g., "  simp"
        # If it's complex like "  have h := by simp", this simple replacement might break syntax.
        # But for "by tactic", it works.
        
        # Check if line starts with 'by ' or is inside a 'by' block
        # For now, just replace the tactic name if it appears cleanly
        
        new_line = failing_line # Default
        
        if stripped_line in ["linarith", "ring", "simp", "rfl", "omega", "aesop"]:
             new_line = f"{indentation}{new_tactic}"
        else:
            # Try substituting the keyword
            # e.g. "simp [add_comm]" -> "aesop [add_comm]" (risky for arguments)
            # Safer: "simp [add_comm]" -> "aesop" (drop arguments)
             new_line = f"{indentation}{new_tactic}"

        if new_line == failing_line:
            continue
            
        print(f"  [AUTO-REPAIR] Attempting to replace '{stripped_line}' with '{new_tactic}' on line {line_idx+1}...")
        
        lines[line_idx] = new_line
        candidate_code = "\n".join(lines)
        
        # Verify
        result = verify_lean(candidate_code, filename=filename)
        
        if result.success:
            print(f"  [AUTO-REPAIR] Success! Repaired with '{new_tactic}'.")
            return True, candidate_code, result
            
        # Revert for next attempt
        lines[line_idx] = failing_line
        
    return False, code_content, None