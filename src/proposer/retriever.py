import json
import os
import shutil
import string
from src.library.indexer import index_modules, INDEX_FILE, DEFAULT_MODULES_DIR
from src.utils.interface_extractor import extract_lean_interface, extract_tla_interface

class Retriever:
    def __init__(self, modules_dir=DEFAULT_MODULES_DIR, index_file=INDEX_FILE):
        self.modules_dir = modules_dir
        self.index_file = index_file
        
        # Load or create index
        if os.path.exists(self.index_file):
            with open(self.index_file, "r") as f:
                self.index = json.load(f)
        else:
            self.index = index_modules(modules_dir=self.modules_dir, output_file=self.index_file)

    def retrieve(self, query: str) -> str:
        """
        Selects relevant modules based on the query and returns their interfaces.
        """
        # Ensure we have an index
        if not self.index:
            return ""

        selected_modules = self._select_modules(query)
        if not selected_modules:
            return ""

        context_blocks = ["\nCONTEXT - AVAILABLE FORMAL TOOLS:"]
        context_blocks.append("-" * 50)

        for mod in selected_modules:
            path = mod['path']
            if not os.path.exists(path):
                continue

            with open(path, "r") as f:
                content = f.read()

            if mod['type'] == 'lean':
                interface = extract_lean_interface(content)
            else:
                interface = extract_tla_interface(content)

            context_blocks.append(f"[MODULE: {mod['id']}]")
            context_blocks.append(f"-- Description: {mod['description']}")
            context_blocks.append(interface)
            context_blocks.append("-" * 50)
            
        if len(context_blocks) <= 2: # Only header
            return ""
            
        return "\n".join(context_blocks)

    def _select_modules(self, query: str):
        """
        Simple keyword-based selection.
        """
        # Normalize and strip punctuation
        translator = str.maketrans('', '', string.punctuation)
        query_clean = query.lower().translate(translator)
        query_words = set(query_clean.split())
        
        # Common stop words
        common = {'the', 'a', 'an', 'of', 'for', 'in', 'on', 'to', 'and', 'is', 'using', 'with'}
        query_words -= common
        
        selected = []
        
        for mod in self.index:
            # Check ID (e.g., "Math" in query)
            mod_id_parts = mod['id'].lower().split('.')
            if any(part in query_clean for part in mod_id_parts):
                selected.append(mod)
                continue
            
            # Check description words
            desc_clean = mod['description'].lower().translate(translator)
            desc_words = set(desc_clean.split()) - common
            
            if desc_words & query_words:
                selected.append(mod)
                
        return selected
