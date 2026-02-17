import unittest
from src.verifiers.skill_verifier import inject_runtime_checks

class TestSkillVerifier(unittest.TestCase):
    def test_injection(self):
        input_code = """
def main():
    x = 10
    y = 5
    res = calculator_add(x, y)
    print(res)
"""
        skill_defs = ["skills/examples/calculator.yaml"]
        
        output_code = inject_runtime_checks(input_code, skill_defs)
        
        # Check for wrapper definition
        self.assertIn("def verified_calculator_add(a, b):", output_code)
        
        # Check for preconditions
        self.assertIn("assert a > 0", output_code)
        
        # Check for postconditions
        self.assertIn("assert result == a + b", output_code)
        
        # Check for replacement
        self.assertIn("res = verified_calculator_add(x, y)", output_code)
        self.assertNotIn("res = calculator_add(x, y)", output_code)

if __name__ == "__main__":
    unittest.main()
