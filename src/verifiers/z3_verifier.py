import subprocess
import sys
from .common import VerificationResult

def verify_z3(script_content: str) -> VerificationResult:
    """
    Executes a Z3 Python script and returns the result.
    The script is expected to use the 'z3' library.
    """
    temp_file = "work/temp_z3.py"
    with open(temp_file, "w") as f:
        f.write(script_content)
    
    try:
        # Use the current python executable (which should have z3-solver installed)
        result = subprocess.run([sys.executable, temp_file], capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            return VerificationResult(True, "Z3 script executed successfully", result.stdout)
        else:
            return VerificationResult(False, "Z3 script failed", result.stderr)
    except subprocess.TimeoutExpired:
        return VerificationResult(False, "Z3 script timed out")
    except Exception as e:
        return VerificationResult(False, f"Error executing Z3 script: {str(e)}")
