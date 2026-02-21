import subprocess
import os
import re
from .common import VerificationResult

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
    
    # Create a default config file if it doesn't exist
    if not os.path.exists(cfg_file):
        with open(cfg_file, "w") as f:
            f.write("SPECIFICATION Spec\n")
            # Only add TypeOK if it is DEFINED (e.g., TypeOK == ...)
            if re.search(r"TypeOK\s*==", spec_content):
                f.write("INVARIANT TypeOK\n")

    try:
        # Use local JDK
        java_bin = "work/jdk/bin/java"
        if not os.path.exists(java_bin):
            java_bin = "java" # fallback
            
        # java -cp tla2tools.jar tlc2.TLC temp.tla
        cmd = [java_bin, "-cp", jar_path, "tlc2.TLC", "-deadlock", tla_file]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        
        if "Model checking completed. No error has been found." in result.stdout:
            return VerificationResult(True, "TLA+ verification successful", result.stdout)
        else:
            return VerificationResult(False, "TLA+ verification failed or found errors", result.stdout + result.stderr)
            
    except Exception as e:
        return VerificationResult(False, f"Error running TLC: {str(e)}")
