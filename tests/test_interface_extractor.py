import unittest
from src.utils.interface_extractor import extract_lean_interface, extract_tla_interface

class TestInterfaceExtractor(unittest.TestCase):
    def test_lean_interface_extraction(self):
        lean_content = """
import Mathlib

/-- A simple math theorem -/
theorem add_comm (n m : Nat) : n + m = m + n := by
  simp [Nat.add_comm]

def add_one (n : Nat) : Nat := n + 1

#eval add_one 5
"""
        expected_interface = """import Mathlib
/-- A simple math theorem -/
theorem add_comm (n m : Nat) : n + m = m + n
def add_one (n : Nat) : Nat"""

        extracted = extract_lean_interface(lean_content)
        # Check specific components
        self.assertIn("theorem add_comm (n m : Nat) : n + m = m + n", extracted)
        self.assertNotIn(":= by", extracted)
        self.assertIn("def add_one (n : Nat) : Nat", extracted)
        self.assertNotIn(":= n + 1", extracted)
        self.assertNotIn("#eval add_one 5", extracted)

    def test_tla_interface_extraction(self):
        tla_content = """
---- MODULE Test ----
EXTENDS Naturals
CONSTANT N
VARIABLE x

Init == x = 0
Next == x' = x + 1

Spec == Init /\ [][Next]_x
====
"""
        expected_interface = """---- MODULE Test ----
EXTENDS Naturals
CONSTANT N
VARIABLE x
Init == x = 0
Next == x' = x + 1
Spec == Init /\ [][Next]_x
====
"""
        extracted = extract_tla_interface(tla_content)
        # Check specific components
        self.assertIn("CONSTANT N", extracted)
        self.assertIn("VARIABLE x", extracted)
        self.assertIn("Init == x = 0", extracted)
        self.assertIn("Next == x' = x + 1", extracted)
        self.assertIn("Spec == Init /\ [][Next]_x", extracted)

if __name__ == "__main__":
    unittest.main()
