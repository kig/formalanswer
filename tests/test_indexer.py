import unittest
import os
import shutil
import json
from src.library.indexer import index_modules

class TestIndexer(unittest.TestCase):
    def setUp(self):
        # Create a temporary modules directory for testing
        self.test_dir = "modules_test"
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
        os.makedirs(self.test_dir)
        
        # Define output file path
        self.output_file = "test_library_index.json"

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
        if os.path.exists(self.output_file):
            os.remove(self.output_file)

    def test_module_indexing(self):
        # Create a dummy Lean module
        math_dir = os.path.join(self.test_dir, "math")
        os.makedirs(math_dir)
        lean_path = os.path.join(math_dir, "probabilities.lean")
        with open(lean_path, "w") as f:
            f.write("/-- Standard probability bounds. -/\ntheorem chebyshev ...")

        # Create a dummy TLA module
        safety_dir = os.path.join(self.test_dir, "safety")
        os.makedirs(safety_dir)
        tla_path = os.path.join(safety_dir, "circuit_breaker.tla")
        with open(tla_path, "w") as f:
            f.write("(* Safety protocol for circuit breakers. *)\nMODULE CircuitBreaker ...")

        # Run the indexer with the test directory and output file
        index = index_modules(modules_dir=self.test_dir, library_dir=None, static_index_file=self.output_file)
            
        # Verify results
        self.assertEqual(len(index), 2)
            
        # Find the math module
        # Note: generate_module_id capitalizes parts. 'math' -> 'Math', 'probabilities' -> 'Probabilities'
        math_mod = next((m for m in index if m["id"] == "Math.Probabilities"), None)
        self.assertIsNotNone(math_mod, "Math.Probabilities module not found")
        self.assertIn("Standard probability bounds", math_mod["description"])
        self.assertEqual(math_mod["type"], "lean")
            
        # Find the safety module
        safety_mod = next((m for m in index if m["id"] == "Safety.Circuit_breaker"), None)
        self.assertIsNotNone(safety_mod, "Safety.Circuit_breaker module not found")
        self.assertIn("Safety protocol", safety_mod["description"])
        self.assertEqual(safety_mod["type"], "tla")
            
        # Check JSON file existence
        self.assertTrue(os.path.exists(self.output_file))

if __name__ == "__main__":
    unittest.main()