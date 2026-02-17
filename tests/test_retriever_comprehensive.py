import unittest
import os
import shutil
import json
from src.proposer.retriever import Retriever
from src.library.indexer import index_modules, INDEX_FILE, DEFAULT_MODULES_DIR
from src.proposer.prompts import format_user_prompt

class TestRetrieverComprehensive(unittest.TestCase):
    def setUp(self):
        # 1. Setup Environment
        self.test_dir = "modules_test_suite"
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
        os.makedirs(self.test_dir)
        
        self.index_file = "test_suite_index.json"
        
        # 2. Populate Library with Realistic Modules
        
        # Module A: Math.Probabilities (Lean)
        math_dir = os.path.join(self.test_dir, "math")
        os.makedirs(math_dir)
        with open(os.path.join(math_dir, "probabilities.lean"), "w") as f:
            f.write("""/-- 
Standard probability bounds (Markov, Chebyshev). 
Use for bounding tail risks.
-/
constant P (event : Set Ω) : Real
theorem markov_bound (X : Var) (a : Real) : P (X ≥ a) ≤ E[X] / a := by sorry
def standard_deviation (X : Var) : Real := sorry
""")

        # Module B: Finance.Volatility (Lean)
        fin_dir = os.path.join(self.test_dir, "finance")
        os.makedirs(fin_dir)
        with open(os.path.join(fin_dir, "volatility.lean"), "w") as f:
            f.write("""/-- 
Volatility models and hedging predicates.
Keywords: finance, market, hedge, crash.
-/
def Volatility (prices : Series) : Real
theorem high_vol_risk : ∀ p, Volatility p > 5 → Risk p := by sorry
""")

        # Module C: Sys.Consensus (TLA+)
        sys_dir = os.path.join(self.test_dir, "sys")
        os.makedirs(sys_dir)
        with open(os.path.join(sys_dir, "consensus.tla"), "w") as f:
            f.write("""(* 
Consensus algorithms (Paxos/Raft). 
Ensures safety and liveness in distributed systems. 
*)
---- MODULE Consensus ----
CONSTANT Nodes
VARIABLE votes
Safe == \A n \in Nodes : votes[n] <= 1
=========================
""")

        # 3. Generate Index
        index_modules(modules_dir=self.test_dir, output_file=self.index_file)
        
        # 4. Initialize Retriever
        self.retriever = Retriever(modules_dir=self.test_dir, index_file=self.index_file)

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
        if os.path.exists(self.index_file):
            os.remove(self.index_file)

    def test_single_module_retrieval(self):
        """Test retrieving a single specific module."""
        query = "Calculate standard deviation using probability bounds."
        context = self.retriever.retrieve(query)
        
        self.assertIn("[MODULE: Math.Probabilities]", context)
        self.assertIn("theorem markov_bound", context)
        self.assertNotIn("[MODULE: Finance.Volatility]", context)

    def test_multi_module_retrieval(self):
        """Test retrieving multiple modules based on mixed keywords."""
        query = "Design a consensus algorithm that handles market volatility."
        context = self.retriever.retrieve(query)
        
        self.assertIn("[MODULE: Sys.Consensus]", context)
        self.assertIn("CONSTANT Nodes", context)
        
        self.assertIn("[MODULE: Finance.Volatility]", context)
        self.assertIn("def Volatility", context)
        
        # Math shouldn't be here (unless "market" triggers it via some shared term, 
        # but based on current simple keywords it shouldn't)
        self.assertNotIn("[MODULE: Math.Probabilities]", context)

    def test_interface_cleanliness(self):
        """Ensure implementation details (proofs) are stripped."""
        query = "probability"
        context = self.retriever.retrieve(query)
        
        self.assertIn("theorem markov_bound", context)
        # "by sorry" should be stripped by the interface extractor
        self.assertNotIn(":= by sorry", context) 

    def test_end_to_end_prompt_construction(self):
        """Simulate the full prompt generation flow."""
        query = "Analyze the risk of a market crash using probability bounds."
        context = self.retriever.retrieve(query)
        
        full_prompt = format_user_prompt(query, context)
        
        # Verify the prompt structure
        self.assertIn("CONTEXT - AVAILABLE FORMAL TOOLS:", full_prompt)
        self.assertIn("[MODULE: Math.Probabilities]", full_prompt)
        self.assertIn("[MODULE: Finance.Volatility]", full_prompt) # "Risk" matches finance desc keywords
        self.assertIn("QUESTION: Analyze the risk", full_prompt)
        self.assertIn("# Mode Selection", full_prompt)

    def test_no_match_behavior(self):
        """Ensure irrelevant queries get clean prompts."""
        query = "Write a poem about a cat."
        context = self.retriever.retrieve(query)
        
        self.assertEqual(context, "")
        
        full_prompt = format_user_prompt(query, context)
        self.assertNotIn("CONTEXT - AVAILABLE FORMAL TOOLS", full_prompt)

if __name__ == "__main__":
    unittest.main()
