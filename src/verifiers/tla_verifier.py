import subprocess
import os
import re
from .common import VerificationResult

def parse_tla_errors(output: str) -> str:
    """
    Parses raw TLC output into a concise, actionable format.
    Extracts: Error type, location, and counter-example trace.
    """
    errors = []
    lines = output.split('\n')
    
    # regex for parser errors: "line 10, col 20 to line 10, col 25 of module temp"
    parser_error_pattern = re.compile(r"line (\d+), col (\d+) to line \d+, col \d+ of module (\w+)")
    
    trace_mode = False
    trace_lines = []
    
    for line in lines:
        # Capture standard errors
        if line.startswith("Error:"):
            errors.append(f"[TLA ERROR] {line.strip()}")
            if "violated" in line:
                trace_mode = True # Start capturing the trace
        
        # Capture specific parser location info
        match = parser_error_pattern.search(line)
        if match:
             errors.append(f"[PARSER ERROR] Line {match.group(1)}, Col {match.group(2)} in module {match.group(3)}")

        # Capture trace (limit to 20 lines to avoid token explosion)
        if trace_mode and len(trace_lines) < 20:
             trace_lines.append(line)
    
    if trace_lines:
        errors.append("\n[COUNTER-EXAMPLE TRACE]:\n" + "\n".join(trace_lines))
        
    if not errors:
        # If no explicit "Error:" found but it failed, return tail
        return "\n".join(lines[-20:])
        
    return "\n".join(errors)

def verify_tla(spec_content: str, module_name: str = "temp") -> VerificationResult:
    """
    Runs the TLC model checker on a TLA+ specification.
    """
    tla_file = f"work/{module_name}.tla"
    cfg_file = f"work/{module_name}.cfg"
    jar_path = "work/tla2tools.jar"

    if not os.path.exists(jar_path):
        return VerificationResult(False, "tla2tools.jar not found. Run setup_manager first.")

    # Force the module name to be 'temp' to match the filename
    spec_content = re.sub(r"MODULE \w+", "MODULE temp", spec_content, count=1)
    
    with open(tla_file, "w") as f:
        f.write(spec_content)
    
    # Always rewrite config file to ensure it matches the current spec
    with open(cfg_file, "w") as f:
        f.write("SPECIFICATION Spec\n")
        # Only add TypeOK if it is DEFINED (e.g., TypeOK == ...)
        if re.search(r"TypeOK\s*==", spec_content):
            f.write("INVARIANT TypeOK\n")

    try:
        # Use local JDK
        java_bin = "work/jdk/Contents/Home/bin/java"
        if not os.path.exists(java_bin):
            java_bin = "java" # fallback
            
        # java -cp tla2tools.jar tlc2.TLC temp.tla
        cmd = [java_bin, "-cp", jar_path, "tlc2.TLC", "-deadlock", tla_file]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        
        if "Model checking completed. No error has been found." in result.stdout:
            return VerificationResult(True, "TLA+ verification successful", result.stdout)
        else:
            # Parse errors from both stdout and stderr
            full_output = result.stdout + "\n" + result.stderr
            parsed_errors = parse_tla_errors(full_output)
            print(f"TLA+ Error Output (Parsed):\n{parsed_errors}")
            return VerificationResult(False, "TLA+ verification failed", parsed_errors)
            
    except Exception as e:
        return VerificationResult(False, f"Error running TLC: {str(e)}")
