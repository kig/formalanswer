import os
import json
import re

DEFAULT_MODULES_DIR = "modules"
INDEX_FILE = "library_index.json"

def generate_module_id(rel_path):
    """
    Converts 'math/probabilities.lean' -> 'Math.Probabilities'
    """
    path_no_ext = os.path.splitext(rel_path)[0]
    parts = path_no_ext.split(os.sep)
    return ".".join(p.capitalize() for p in parts)

def extract_description(content, file_ext):
    """
    Extracts the first docstring block from the file content.
    """
    if file_ext == ".lean":
        match = re.search(r'/--\s*(.*?)\s*-/', content, re.DOTALL)
        if match:
            return match.group(1).strip()
    elif file_ext == ".tla":
        match = re.search(r'\(\*\s*(.*?)\s*\*\)', content, re.DOTALL)
        if match:
            return match.group(1).strip()
    return "No description provided."

def index_modules(modules_dir=DEFAULT_MODULES_DIR, library_dir="library", output_file=INDEX_FILE):
    """
    Scans modules/ (hand-written) and library/ (successful tasks) to generate library_index.json.
    """
    index = []
    
    # 1. Scan Hand-Written Modules
    if os.path.exists(modules_dir):
        for root, dirs, files in os.walk(modules_dir):
            for file in files:
                if file.endswith((".lean", ".tla")):
                    full_path = os.path.join(root, file)
                    rel_path = os.path.relpath(full_path, modules_dir)
                    
                    with open(full_path, "r") as f:
                        content = f.read()
                    
                    module_id = generate_module_id(rel_path)
                    description = extract_description(content, os.path.splitext(file)[1])
                    
                    entry = {
                        "id": module_id,
                        "path": full_path,
                        "description": description,
                        "type": "lean" if file.endswith(".lean") else "tla"
                    }
                    index.append(entry)

    # 2. Scan Successful Task Library
    if os.path.exists(library_dir):
        # library/TaskName/proof_candidate.lean
        for task_name in os.listdir(library_dir):
            task_path = os.path.join(library_dir, task_name)
            if not os.path.isdir(task_path):
                continue
                
            # Check for success indicators
            has_lean = os.path.exists(os.path.join(task_path, "proof_candidate.lean"))
            has_tla = os.path.exists(os.path.join(task_path, "proof_candidate.tla"))
            
            if not (has_lean or has_tla):
                continue
                
            # Get description from prompt.txt
            description = "Successfully verified task."
            prompt_path = os.path.join(task_path, "prompt.txt")
            if os.path.exists(prompt_path):
                with open(prompt_path, "r") as f:
                    description = f.read().strip()[:200].replace("\n", " ") # First 200 chars
            
            # Add Lean module
            if has_lean:
                entry = {
                    "id": f"Task.{task_name}.Lean",
                    "path": os.path.join(task_path, "proof_candidate.lean"),
                    "description": description,
                    "type": "lean"
                }
                index.append(entry)
                
            # Add TLA module
            if has_tla:
                entry = {
                    "id": f"Task.{task_name}.TLA",
                    "path": os.path.join(task_path, "proof_candidate.tla"),
                    "description": description,
                    "type": "tla"
                }
                index.append(entry)

    with open(output_file, "w") as f:
        json.dump(index, f, indent=2)
        
    print(f"Indexed {len(index)} modules to {output_file}")
    return index

if __name__ == "__main__":
    index_modules()