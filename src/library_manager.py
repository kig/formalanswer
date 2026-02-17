import os
import time

class LibraryManager:
    def __init__(self, library_dir="library"):
        self.library_dir = library_dir
        if not os.path.exists(self.library_dir):
            os.makedirs(self.library_dir)

    def init_task_dir(self, task_name):
        """
        Creates a new directory for the current task session.
        """
        timestamp = int(time.time())
        # Sanitize task name
        safe_name = "".join(x for x in task_name if x.isalnum() or x in " -_").strip().replace(" ", "_")[:50]
        base_dir_name = f"{safe_name}_{timestamp}"
        target_dir = os.path.join(self.library_dir, base_dir_name)
        
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
            
        raw_dir = os.path.join(target_dir, "raw")
        if not os.path.exists(raw_dir):
            os.makedirs(raw_dir)
            
        return target_dir

    def save_raw_response(self, task_dir, iteration_idx, response, label=None):
        """
        Saves a raw LLM response to the task directory.
        """
        raw_dir = os.path.join(task_dir, "raw")
        filename = f"{iteration_idx}.txt"
        if label:
            filename = f"{iteration_idx}_{label}.txt"
            
        path = os.path.join(raw_dir, filename)
        with open(path, "w") as f:
            f.write(response)
        return path

    def save_proofs(self, task_dir, blocks):
        """
        Saves the successful proofs to the task directory.
        """
        saved_files = []

        if blocks.get("tla"):
            path = os.path.join(task_dir, "proof.tla")
            with open(path, "w") as f:
                f.write(blocks["tla"])
            saved_files.append(path)

        if blocks.get("python"):
            path = os.path.join(task_dir, "proof.py")
            with open(path, "w") as f:
                f.write(blocks["python"])
            saved_files.append(path)

        if blocks.get("lean"):
            path = os.path.join(task_dir, "proof.lean")
            with open(path, "w") as f:
                f.write(blocks["lean"])
            saved_files.append(path)

        if blocks.get("prose"):
            path = os.path.join(task_dir, "proof.txt")
            with open(path, "w") as f:
                f.write(blocks["prose"])
            saved_files.append(path)

        print(f"\n[LIBRARY] Saved successful proofs to {task_dir}/")
        return saved_files
