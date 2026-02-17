import unittest
import os
import sys
from unittest.mock import MagicMock

# Ensure src is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.proposer.client import Proposer
from src.verifiers.python_verifier import verify_python

class TestJaxMock(unittest.TestCase):
    def setUp(self):
        self.proposer = Proposer(backend="gemini", api_key="dummy_key")
        self.proposer.chat = MagicMock()

    def test_jax_simulation_execution(self):
        """
        Test that a JAX-based simulation script can be extracted and executed successfully.
        MANDATORY: Requires JAX installed in the environment.
        """
        # A mock response containing a valid JAX script
        mock_response = """
# Mode Selection
[MODE: PROBABILISTIC]

# Z3/Python Script (The Empirical Grounding)
```python
import jax.numpy as jnp
from jax import random, vmap

def simulation_kernel(key):
    return jnp.where(random.uniform(key) > 0.5, 1.0, 0.0)

def run_simulation():
    key = random.PRNGKey(42)
    keys = random.split(key, 1000)
    results = vmap(simulation_kernel)(keys)
    mean_val = jnp.mean(results)
    print(f"Simulation Mean: {mean_val}")
    assert 0.4 < mean_val < 0.6

if __name__ == "__main__":
    run_simulation()
```
"""
        # 1. Extract the code
        blocks = self.proposer.extract_code(mock_response)
        self.assertIsNotNone(blocks["python"])

        # 2. Execute the script using the verifier
        # This will now fail if jax is not installed in the subprocess environment
        result = verify_python(blocks["python"])
        
        self.assertTrue(result.success, f"JAX script execution failed: {result.message}\\nDetails: {result.details}")
        self.assertIn("Simulation Mean:", result.details)

if __name__ == "__main__":
    unittest.main()
