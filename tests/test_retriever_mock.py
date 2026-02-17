import unittest
import os
import shutil
import json
import re

# Need to manually construct the index if we want to test Retriever in isolation,
# or we can rely on indexer. But let's verify Retriever logic specifically.

from src.proposer.retriever import Retriever
from src.library.indexer import index_modules

class TestRetriever(unittest.TestCase):
    def setUp(self):
        # Temp dir for modules
        self.test_dir = "modules_test_retriever"
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
        os.makedirs(self.test_dir)
        
        # Temp index file
        self.index_file = "test_retriever_index.json"

        # Create dummy module
        math_dir = os.path.join(self.test_dir, "math")
        os.makedirs(math_dir)
        lean_path = os.path.join(math_dir, "probabilities.lean")
        
        with open(lean_path, "w") as f:
            f.write("""/-- Standard probability bounds. -/
theorem chebyshev (x : Nat) : x > 0 := by simp""")

        # Generate index
        index_modules(modules_dir=self.test_dir, output_file=self.index_file)

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
        if os.path.exists(self.index_file):
            os.remove(self.index_file)

    def test_retrieve_match(self):
        # Initialize Retriever
        retriever = Retriever(modules_dir=self.test_dir, index_file=self.index_file)
        
        # Query matching "probability" description
        query = "Verify probability bounds using chebyshev."
        context = retriever.retrieve(query)
        
        # Verify Context Structure
        self.assertIn("CONTEXT - AVAILABLE FORMAL TOOLS:", context)
        self.assertIn("[MODULE: Math.Probabilities]", context)
        
        # Verify Interface Extraction (Proof should be stripped)
        self.assertIn("theorem chebyshev (x : Nat) : x > 0", context)
        self.assertNotIn(":= by simp", context)

    def test_retrieve_no_match(self):
        retriever = Retriever(modules_dir=self.test_dir, index_file=self.index_file)
        query = "Design a completely unrelated system."
        context = retriever.retrieve(query)
        
        # Should be empty string if no match
        self.assertEqual(context, "")

if __name__ == "__main__":
    unittest.main()