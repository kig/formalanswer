import subprocess
import os
import re
from .common import VerificationResult
from .lean_server import get_lean_server

def parse_lean_errors(output: str) -> str:
    # ... (existing parse_lean_errors remains same) ...
    """
    Parses raw Lean stderr output into a concise, actionable format for the LLM.
    Extracts: Line number, Column number, Error message.
    """
    errors = []
    lines = output.split('\n')
    
    # Regex for standard Lean error format: "filename:line:col: error: message"
    # Example: "temp.lean:41:2: error: linarith failed to find a contradiction"
    error_pattern = re.compile(r"^.*:(\d+):(\d+):\s+(error|warning):\s+(.*)$")
    
    current_error = None
    
    for line in lines:
        match = error_pattern.match(line)
        if match:
            # If we were tracking a previous error, save it
            if current_error:
                errors.append(current_error)
            
            line_num = match.group(1)
            col_num = match.group(2)
            severity = match.group(3).upper()
            msg = match.group(4)
            current_error = f"[{severity}] Line {line_num}, Col {col_num}: {msg}"
        elif current_error and line.strip() and not line.startswith("error:"):
            # Append context lines (e.g., the goal state or variables involved)
            # Limit context to avoid flooding the prompt
            if len(current_error.split('\n')) < 5: 
                current_error += f"\n    {line.strip()}"
    
    if current_error:
        errors.append(current_error)
        
    if not errors:
        return output # Fallback to raw output if parsing fails
        
    return "\n".join(errors)

def verify_lean(code_content: str, filename: str = "temp.lean") -> VerificationResult:
    """
    Runs the Lean 4 compiler on a .lean file.
    Uses persistent LeanServer if possible, fallbacks to lake env lean.
    """
    if os.environ.get("DISABLE_LEAN_SERVER") != "1":
        try:
            server = get_lean_server()
            result = server.verify_snippet(code_content)
            # If server passed, or failed with specific logical errors, trust it.
            # If it's a server-level failure, we fallback.
            if result.success or "failed" in result.message:
                return result
        except Exception as e:
            print(f"Lean Server failed, falling back to subprocess: {e}")

    # Fallback to subprocess execution
    lean_file = f"work/{filename}"
    with open(lean_file, "w") as f:
        f.write(code_content)
    
    try:
        # Run lean via lake to access dependencies
        # cwd="work" ensures we are in the lake project root
        # Use env_wrapper.sh to ensure local elan/lean are in PATH
        cmd = ["../env_wrapper.sh", "lake", "env", "lean", filename]
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
            # Parse stderr for specific error messages
            parsed_errors = parse_lean_errors(result.stderr)
            print(f"Lean Error Output (Parsed):\n{parsed_errors}")
            return VerificationResult(False, "Lean verification failed", parsed_errors)
            
    except Exception as e:
        return VerificationResult(False, f"Error running Lean: {str(e)}")
