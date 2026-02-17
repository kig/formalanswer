import subprocess
import sys
import os
from .common import VerificationResult

def verify_python(script_content: str) -> VerificationResult:
    """
    Executes a Python script (Z3 constraints or Monte Carlo simulation) and returns the verification result.
    
    A return code of 0 implies success (verification passed).
    A non-zero return code (e.g., assertion error, exception) implies failure.
    
    The script is executed in a temporary file: work/temp_script.py
    """
    if not os.path.exists("work"):
        os.makedirs("work")
        
    temp_file = "work/temp_script.py"
    with open(temp_file, "w") as f:
        f.write(script_content)
    
    try:
        # Use the current python executable.
        # Increased timeout to 60s for simulations.
        result = subprocess.run(
            [sys.executable, temp_file], 
            capture_output=True, 
            text=True, 
            timeout=60
        )
        
        if result.returncode == 0:
            # Script ran successfully.
            return VerificationResult(True, "Python script executed successfully", result.stdout)
        else:
            # Script failed (e.g., assertion error).
            return VerificationResult(False, "Python script failed", f"Error Output:\n{result.stderr}\nStandard Output:\n{result.stdout}")
            
    except subprocess.TimeoutExpired:
        return VerificationResult(False, "Python script timed out (Limit: 60s)")
    except Exception as e:
        return VerificationResult(False, f"Error executing Python script: {str(e)}")