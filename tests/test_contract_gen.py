import unittest
import os
from src.skills.contract_generator import generate_contracts

class TestContractGenerator(unittest.TestCase):
    def test_calculator_contract(self):
        skill_path = "skills/examples/calculator.yaml"
        lean, tla = generate_contracts(skill_path)
        
        # Verify Lean
        self.assertIn("constant calculator_add", lean)
        self.assertIn("axiom calculator_add_spec", lean)
        self.assertIn("a > 0", lean) # Precondition
        self.assertIn("result == a + b", lean) # Postcondition
        
        # Verify TLA+
        self.assertIn("calculator_addPre(a, b)", tla)
        self.assertIn("calculator_addPost(a, b, result)", tla)
        self.assertIn("a > 0", tla)

if __name__ == "__main__":
    unittest.main()
