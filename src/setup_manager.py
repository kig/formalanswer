import subprocess
import os
import sys
import shutil
import urllib.request

class SetupManager:
    def __init__(self, work_dir="work"):
        self.work_dir = work_dir
        if not os.path.exists(self.work_dir):
            os.makedirs(self.work_dir)
        self.tla2tools_path = os.path.join(self.work_dir, "tla2tools.jar")

    def check_command(self, cmd):
        return shutil.which(cmd) is not None

    def run_command(self, cmd):
        try:
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            return result.returncode == 0, result.stdout, result.stderr
        except Exception as e:
            return False, "", str(e)

    def setup_z3(self):
        print("Checking Z3...")
        try:
            import z3
            print(f"Z3 version: {z3.get_version_string()}")
            return True
        except ImportError:
            print("Z3 not found. Attempting to install...")
            success, out, err = self.run_command("pip install z3-solver")
            if success:
                print("Z3 installed successfully.")
                return True
            else:
                print(f"Failed to install Z3: {err}")
                return False

    def setup_lean(self):
        print("Checking Lean 4...")
        if self.check_command("lean"):
            success, out, err = self.run_command("lean --version")
            print(f"Lean version: {out.strip()}")
            return True
        else:
            print("Lean 4 not found. Please install elan/lean4.")
            # We won't attempt to install lean automatically as it's a heavy process
            return False

    def setup_java(self):
        print("Checking Java...")
        if self.check_command("java"):
            success, out, err = self.run_command("java -version")
            # java -version outputs to stderr
            print("Java is installed.")
            return True
        else:
            print("Java not found. TLA+ requires Java.")
            return False

    def setup_tla2tools(self):
        print("Checking tla2tools.jar...")
        if os.path.exists(self.tla2tools_path):
            print(f"tla2tools.jar found at {self.tla2tools_path}")
            return True
        else:
            print("tla2tools.jar not found. Downloading...")
            url = "https://github.com/tlaplus/tlaplus/releases/download/v1.8.0/tla2tools.jar"
            try:
                urllib.request.urlretrieve(url, self.tla2tools_path)
                print("Downloaded tla2tools.jar successfully.")
                return True
            except Exception as e:
                print(f"Failed to download tla2tools.jar: {e}")
                return False

    def setup_all(self):
        results = {
            "z3": self.setup_z3(),
            "lean": self.setup_lean(),
            "java": self.setup_java(),
            "tla2tools": self.setup_tla2tools()
        }
        return results

if __name__ == "__main__":
    manager = SetupManager()
    results = manager.setup_all()
    print("\nSetup Summary:")
    for tool, status in results.items():
        print(f"{tool}: {'OK' if status else 'MISSING'}")
    
    if not all(results.values()):
        sys.exit(1)
