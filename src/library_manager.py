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

    def save_prompt(self, task_dir, prompt_content):
        """
        Saves the original prompt to the task directory.
        """
        path = os.path.join(task_dir, "prompt.txt")
        with open(path, "w") as f:
            f.write(prompt_content)
        return path

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

    def save_candidate_proofs(self, task_dir, blocks):
        """
        Saves the current candidate proofs to the task directory.
        Handles lists of blocks.
        """
        self._save_blocks(task_dir, blocks, prefix="proof_candidate")

    def save_proofs(self, task_dir, blocks, original_prompt=None):
        """
        Saves the successful proofs to the task directory.
        Optionally saves the original prompt.
        """
        saved_files = self._save_blocks(task_dir, blocks, prefix="proof")

        if blocks.get("prose"):
            path = os.path.join(task_dir, "proof.txt")
            with open(path, "w") as f:
                f.write(blocks["prose"])
            saved_files.append(path)
            
        if original_prompt:
            self.save_prompt(task_dir, original_prompt)

        print(f"\n[LIBRARY] Saved successful proofs to {task_dir}/")
        return saved_files

    def _save_blocks(self, task_dir, blocks, prefix="proof"):
        """
        Helper to save code blocks (which can be lists).
        """
        saved = []
        
        for lang, ext in [("tla", "tla"), ("python", "py"), ("lean", "lean")]:
            content_list = blocks.get(lang)
            if not content_list:
                continue
                
            # Ensure it's a list
            if isinstance(content_list, str):
                content_list = [content_list]
                
            for i, content in enumerate(content_list):
                # If only one block, use standard name. If multiple, use index.
                if len(content_list) == 1:
                    filename = f"{prefix}.{ext}"
                else:
                    filename = f"{prefix}_{i}.{ext}"
                    
                path = os.path.join(task_dir, filename)
                with open(path, "w") as f:
                    f.write(content)
                saved.append(path)
        return saved