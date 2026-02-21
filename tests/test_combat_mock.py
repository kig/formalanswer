import unittest
from unittest.mock import MagicMock, patch
from src.main import FormalReasoningLoop
from src.proposer.client import Proposer

class TestCombatMock(unittest.TestCase):
    @patch("src.controller.verify_tla")
    @patch("src.controller.verify_lean")
    @patch("src.controller.verify_python")
    @patch("src.proposer.client.Proposer") 
    def test_combat_loop(self, MockProposerClass, mock_py, mock_lean, mock_tla):
        # Setup Verifiers to PASS
        mock_tla.return_value = MagicMock(success=True)
        mock_lean.return_value = MagicMock(success=True)
        mock_py.return_value = MagicMock(success=True)
        
        # Setup Proposer Mock
        mock_proposer = MockProposerClass.return_value
        
        # Iteration 1: Propose valid proofs
        # Iteration 2: Propose refined proofs
        mock_proposer.propose.side_effect = [
            "Proof 1 (Weak)", # Iteration 1 Proposal
            "Proof 2 (Strong)", # Iteration 2 Proposal (after combat fail)
            "Final Answer" # Final Analysis
        ]
        
        mock_proposer.extract_code.side_effect = [
            {"tla": ["valid"], "lean": ["valid"], "prose": "Weak Argument"}, # Iteration 1 blocks
            {"tla": ["valid"], "lean": ["valid"], "prose": "Strong Argument"} # Iteration 2 blocks
        ]
        
        # Combat Logic:
        # Iteration 1: Critique finds flaw, Judge gives low score (FAIL)
        # Iteration 2: Critique finds flaw, Judge gives high score (PASS)
        
        mock_proposer.critique.side_effect = ["This is weak.", "This is solid."]
        mock_proposer.judge.side_effect = [(0.2, "Weak commentary"), (0.9, "Strong commentary")] # Fail, then Pass
        
        # Initialize Loop with Combat=True
        loop = FormalReasoningLoop(max_iterations=3, combat=True)
        # We need to manually set the mock proposer into the loop because __init__ creates a new one
        # But we patched the Class, so loop.proposer IS mock_proposer.
        loop.proposer = mock_proposer 
        
        # Run
        success, answer = loop.run("Test Task")
        
        # Assertions
        self.assertTrue(success)
        self.assertIn("Final Answer", answer)
        self.assertIn("# Verified Formal Proofs", answer)
        self.assertIn("## TLA+ Specification", answer)
        self.assertIn("## Lean 4 Proof", answer)
        
        # Verify Critique was called twice
        self.assertEqual(mock_proposer.critique.call_count, 2)
        
        # Verify Judge was called twice
        self.assertEqual(mock_proposer.judge.call_count, 2)

if __name__ == "__main__":
    unittest.main()
