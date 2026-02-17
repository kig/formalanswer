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
        self.assertIsNotNone(blocks["tla"])
        self.assertIn("MODULE temp", blocks["tla"])
        self.assertIsNotNone(blocks["lean"])
        self.assertIn("import Mathlib", blocks["lean"])
        self.assertIsNotNone(blocks["python"])
        self.assertIn("from z3 import *", blocks["python"])
        
        # Verify prose
        self.assertIn("# Rationale", blocks["prose"])
        self.assertNotIn("```tla", blocks["prose"])
        self.assertNotIn("```lean", blocks["prose"])
        self.assertNotIn("```python", blocks["prose"])
        self.assertNotIn("EXTENDS Naturals", blocks["prose"])

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
        # Lean might fail if regex expects closing ticks, let's verify behavior
        # The current regex ````lean\s*\n?(.*?)\n?\s*``` ` expects closing ticks.
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
        self.assertEqual(response, "Mock Response")
        self.proposer.chat.send_message.assert_called_once()

if __name__ == "__main__":
    unittest.main()
