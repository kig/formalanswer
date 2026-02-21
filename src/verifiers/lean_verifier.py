import subprocess
import os
from .common import VerificationResult

def verify_lean(code_content: str) -> VerificationResult:
    """
    Runs the Lean 4 compiler on a .lean file.
    """
    lean_file = "work/temp.lean"
    with open(lean_file, "w") as f:
        f.write(code_content)
    
    try:
        # Run lean via lake to access dependencies
        # cwd="work" ensures we are in the lake project root
        # Use env_wrapper.sh to ensure local elan/lean are in PATH
        cmd = ["../env_wrapper.sh", "lake", "env", "lean", "temp.lean"]
        result = subprocess.run(
            cmd, 
            cwd="work",
            capture_output=True, 
            text=True, 
            timeout=180
        )
        
        if result.returncode == 0:
            # Check for "sorry" which means the proof is incomplete but might "compile"
            if "sorry" in code_content:
                return VerificationResult(False, "Lean code contains 'sorry' (incomplete proof)", result.stdout)
            return VerificationResult(True, "Lean verification successful", result.stdout)
        else:
            # Print stderr for debugging
            print(f"Lean Error Output:\n{result.stderr}")
            return VerificationResult(False, "Lean verification failed", result.stderr)
            
    except Exception as e:
        return VerificationResult(False, f"Error running Lean: {str(e)}")
