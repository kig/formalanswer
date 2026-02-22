import unittest
from unittest.mock import MagicMock, patch
from src.main import FormalReasoningLoop

class TestRapFinalAnalysis(unittest.TestCase):
    @patch("src.controller.verify_tla")
    @patch("src.controller.verify_lean")
    @patch("src.controller.verify_python")
    @patch("src.proposer.client.Proposer") 
    def test_rap_battle_final_analysis(self, MockProposerClass, mock_py, mock_lean, mock_tla):
        # Setup Verifiers to PASS
        mock_tla.return_value = MagicMock(success=True)
        mock_lean.return_value = MagicMock(success=True)
        mock_py.return_value = MagicMock(success=True)
        
        # Setup Proposer Mock
        mock_proposer = MockProposerClass.return_value
        
        mock_proposer.propose.side_effect = [
            "Rap Proposal", # Iteration 1
            "Final Rap Answer" # Final Analysis
        ]
        
        mock_proposer.extract_code.return_value = {
            "tla": ["MODULE temp"], 
            "lean": ["theorem simple"], 
            "prose": "Rap Prose"
        }
        
        # Mock other proposer methods to return strings
        mock_proposer.construct_final_rap.return_value = "Final Track"
        mock_proposer.produce_song_gen_lyrics.return_value = "Song Gen Format"
        
        # Initialize Loop with rap_battle=True
        loop = FormalReasoningLoop(max_iterations=1, rap_battle=True)
        loop.proposer = mock_proposer 
        
        # Run
        success, answer = loop.run("Test Rap Task")
        
        # Assertions
        self.assertTrue(success)
        
        # 1. Check if propose was called for final analysis with rap_battle=True
        # Calls: 1 (Iteration 1) + 1 (Final Analysis) = 2
        self.assertEqual(mock_proposer.propose.call_count, 2)
        
        last_call_args = mock_proposer.propose.call_args
        self.assertTrue(last_call_args.kwargs.get("rap_battle"))
        
        # 2. Check if formal proofs are in the answer
        self.assertIn("Final Rap Answer", answer)
        self.assertIn("# Verified Formal Proofs", answer)
        self.assertIn("## TLA+ Specification", answer)
        self.assertIn("## Lean 4 Proof", answer)

if __name__ == "__main__":
    unittest.main()
