import os
import time

class LibraryManager:
    def __init__(self, library_dir="library"):
        self.library_dir = library_dir
        if not os.path.exists(self.library_dir):
            os.makedirs(self.library_dir)

    def save_proof_set(self, task_name, blocks):
        """
        Saves a successful set of proofs (TLA+, Z3, Lean) to the library.
        """
        timestamp = int(time.time())
        # Sanitize task name
        safe_name = "".join(x for x in task_name if x.isalnum() or x in " -_").strip().replace(" ", "_")[:50]
        base_filename = f"{safe_name}_{timestamp}"

        saved_files = []

        if blocks.get("tla"):
            path = os.path.join(self.library_dir, f"{base_filename}.tla")
            with open(path, "w") as f:
                f.write(blocks["tla"])
            saved_files.append(path)

        if blocks.get("z3"):
            path = os.path.join(self.library_dir, f"{base_filename}.py")
            with open(path, "w") as f:
                f.write(blocks["z3"])
            saved_files.append(path)

        if blocks.get("lean"):
            path = os.path.join(self.library_dir, f"{base_filename}.lean")
            with open(path, "w") as f:
                f.write(blocks["lean"])
            saved_files.append(path)

        print(f"\n[LIBRARY] Saved successful proofs to {self.library_dir}/")
        return saved_files
