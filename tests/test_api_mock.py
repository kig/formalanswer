import unittest
from unittest.mock import MagicMock, patch
from src.api import ProofLoopAPI, VerificationRequest

class TestFormalAPI(unittest.TestCase):
    @patch("src.api.FormalReasoningLoop")
    def test_verify_success(self, MockLoop):
        # Setup Mock
        mock_instance = MockLoop.return_value
        mock_instance.run.return_value = (True, "The system is safe.")
        
        # Test
        api = ProofLoopAPI()
        req = VerificationRequest(task="Verify lock", model="test-model")
        res = api.verify(req)
        
        # Assertions
        MockLoop.assert_called_with(model="test-model", verbose=False)
        mock_instance.run.assert_called_with("Verify lock")
        
        self.assertTrue(res.success)
        self.assertEqual(res.answer, "The system is safe.")

if __name__ == "__main__":
    unittest.main()
