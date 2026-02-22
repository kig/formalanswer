import unittest
from unittest.mock import MagicMock, patch
from src.proposer.client import Proposer

class TestProposerMock(unittest.TestCase):
    def setUp(self):
        # Initialize Proposer with a dummy API key to pass initialization
        # We will mock the client anyway
        self.proposer = Proposer(backend="gemini", api_key="dummy_key")
        self.proposer.chat = MagicMock()

    def test_extract_code_valid(self):
        """Test extraction of valid code blocks."""
        response = """
# Rationale
...

# TLA+ Specification
```tla
----
MODULE temp ----
EXTENDS Naturals
VARIABLE x
Init == x = 0
Next == x' = x + 1
Spec == Init /\ [][Next]_x
====
```

# Lean 4 Proof
```lean
import Mathlib
example : 1 = 1 := by rfl
```

# Z3 Script
```python
from z3 import *
s = Solver()
```
"""
        blocks = self.proposer.extract_code(response)
        
        # TLA Check
        self.assertIsNotNone(blocks["tla"])
        self.assertIsInstance(blocks["tla"], list)
        self.assertEqual(len(blocks["tla"]), 1)
        self.assertIn("MODULE temp", blocks["tla"][0])
        
        # Lean Check
        self.assertIsNotNone(blocks["lean"])
        self.assertIsInstance(blocks["lean"], list)
        self.assertIn("import Mathlib", blocks["lean"][0])
        
        # Python Check
        self.assertIsNotNone(blocks["python"])
        self.assertIsInstance(blocks["python"], list)
        self.assertIn("from z3 import *", blocks["python"][0])
        
        # Verify prose
        self.assertIn("# Rationale", blocks["prose"])
        self.assertNotIn("```tla", blocks["prose"])
        self.assertNotIn("```lean", blocks["prose"])
        self.assertNotIn("```python", blocks["prose"])
        self.assertNotIn("EXTENDS Naturals", blocks["prose"])

    def test_extract_code_multiple(self):
        """Test extraction of MULTIPLE code blocks."""
        response = """
# Plan A
```tla
MODULE A
```
# Plan B
```tla
MODULE B
```
"""
        blocks = self.proposer.extract_code(response)
        self.assertIsNotNone(blocks["tla"])
        self.assertEqual(len(blocks["tla"]), 2)
        self.assertIn("MODULE A", blocks["tla"][0])
        self.assertIn("MODULE B", blocks["tla"][1])

    def test_extract_code_broken(self):
        """Test extraction when code blocks are malformed or missing."""
        response = """
Here is the TLA code:
```tla
MODULE temp
...
```
But I forgot the closing tick for Lean:
```lean
example : 1=1
"""
        blocks = self.proposer.extract_code(response)
        # TLA should be extracted (regex handles simple cases)
        self.assertIsNotNone(blocks["tla"])
        self.assertEqual(len(blocks["tla"]), 1)
        # Lean might fail if regex expects closing ticks
        self.assertIsNone(blocks["lean"])
        self.assertIsNone(blocks["python"])

    def test_extract_code_empty(self):
        """Test extraction from a response with no code."""
        response = "I cannot provide the code."
        blocks = self.proposer.extract_code(response)
        self.assertIsNone(blocks["tla"])
        self.assertIsNone(blocks["lean"])
        self.assertIsNone(blocks["python"])

    def test_propose_mock_call(self):
        """Test that propose calls the chat method."""
        self.proposer.chat.send_message.return_value.text = "Mock Response"
        response = self.proposer.propose("Test Task")
        self.assertEqual(response.content, "Mock Response")
        self.proposer.chat.send_message.assert_called_once()

    def test_propose_rap_battle(self):
        """Test that rap_battle flag adds context prefix."""
        self.proposer.chat.send_message.return_value.text = "Mock Rap Response"
        self.proposer.propose("Test Task", rap_battle=True)
        
        # Check call arguments
        args, _ = self.proposer.chat.send_message.call_args
        prompt = args[0]
        self.assertIn("Logic Rap Battle", prompt)
        self.assertIn("RAP VERSE", prompt)

    def test_propose_repair_rap_battle(self):
        """Test that rap_battle flag is preserved in repair loop."""
        self.proposer.chat.send_message.return_value.text = "Mock Fixed Rap Response"
        self.proposer.propose("Test Task", feedback="Error in Lean", rap_battle=True)
        
        args, _ = self.proposer.chat.send_message.call_args
        prompt = args[0]
        self.assertIn("Logic Rap Battle", prompt)
        self.assertIn("LOGIC BATTLE: REPAIR PROTOCOL", prompt)

    def test_propose_final_analysis_rap_battle(self):
        """Test that rap_battle flag is used in final analysis."""
        self.proposer.chat.send_message.return_value.text = "Mock Final Rap Answer"
        # Simulate final analysis call (task is the analysis prompt)
        self.proposer.propose("Final prompt", rap_battle=True)
        
        args, _ = self.proposer.chat.send_message.call_args
        prompt = args[0]
        self.assertIn("Logic Rap Battle", prompt)

if __name__ == "__main__":
    unittest.main()