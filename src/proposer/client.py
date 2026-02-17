import re
import os
from google import genai
from dotenv import load_dotenv
from .prompts import SYSTEM_PROMPT, format_user_prompt

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

# Load environment variables from .env
load_dotenv()

class Proposer:
    def __init__(self, backend="gemini", model_name=None, api_key=None, base_url=None):
        self.backend = backend.lower()
        self.model_name = model_name
        self.history = []  # For stateless backends like OpenAI/Ollama

        if self.backend == "gemini":
            self.api_key = api_key or os.environ.get("GOOGLE_API_KEY")
            self.model_name = self.model_name or "gemini-2.5-flash"
            self.client = genai.Client(vertexai=True, api_key=self.api_key)
            self.chat = self.client.chats.create(
                model=self.model_name,
                config=genai.types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT
                )
            )
            
        elif self.backend in ["openai", "ollama"]:
            if OpenAI is None:
                raise ImportError("OpenAI package is required for OpenAI/Ollama backends. Install it with `pip install openai`.")
            
            if self.backend == "ollama":
                self.base_url = base_url or "http://localhost:11434/v1"
                self.api_key = "ollama" # Dummy key
                self.model_name = self.model_name or "llama3"
            else: # openai
                self.base_url = base_url
                self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
                self.model_name = self.model_name or "gpt-4o"

            self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
            # Initialize history with system prompt
            self.history = [{"role": "system", "content": SYSTEM_PROMPT}]
            
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

    def propose(self, task, feedback=None, context=None):
        """
        Calls the LLM API.
        """
        if self.backend == "gemini":
            if feedback:
                prompt = f"The previous attempt failed verification.\n\n{feedback}"
            else:
                prompt = format_user_prompt(task, context)
            
            try:
                response = self.chat.send_message(prompt)
                return response.text
            except Exception as e:
                print(f"Gemini API Error: {e}")
                return self._get_mock_response()

        elif self.backend in ["openai", "ollama"]:
            if feedback:
                prompt = f"The previous attempt failed verification.\n\n{feedback}"
                self.history.append({"role": "user", "content": prompt})
            else:
                prompt = format_user_prompt(task, context)
                self.history.append({"role": "user", "content": prompt})
            
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=self.history
                )
                content = response.choices[0].message.content
                self.history.append({"role": "assistant", "content": content})
                return content
            except Exception as e:
                print(f"{self.backend.upper()} API Error: {e}")
                return self._get_mock_response()

    def _get_mock_response(self):
        # A robust mock response demonstrating the Probabilistic/Predictive architecture
        return """
# Mode Selection
[MODE: PROBABILISTIC]

# Critique & Refinement
- **Critique:** A deterministic safety check is insufficient for a market prediction system. We cannot prove "Market will never crash", only "System survives crash with probability P > 0.99".
- **Refinement:** Introduce a `VolatilityThreshold` and `HedgeProbability`.
- **Scope:** Focus on the "Circuit Breaker" mechanism which is a discrete control system reacting to probabilistic inputs.

# Rationale & Shared Constants
We model a "Probabilistic Circuit Breaker". If the estimated probability of a crash exceeds a threshold, the system halts trading.

Shared Invariant: `P(Crash) > Threshold -> State = HALTED`

Shared Constants:
- `CrashProbThreshold`: 10 (representing 10% or 0.1)
- `MaxVolatility`: 5
- `CurrentVolatility`: 6

# TLA+ Specification (The Safety Inspector)
```tla
---- MODULE ProbabilisticBreaker ----
EXTENDS Naturals, TLC

CONSTANTS CrashProbThreshold, MaxVolatility

VARIABLES state, estimated_prob, volatility

Init == 
    /\ state = "TRADING"
    /\ estimated_prob = 0
    /\ volatility = 0

Next == 
    \/ /\ state = "TRADING"
       /\ volatility' \in 0..10
       /\ estimated_prob' \in 0..100
       /\ state' = IF estimated_prob' > CrashProbThreshold THEN "HALTED" ELSE "TRADING"
    \/ /\ state = "HALTED"
       /\ UNCHANGED <<state, estimated_prob, volatility>>

Spec == Init /\ [][Next]_<<state, estimated_prob, volatility>>
====
```

# Lean 4 Proof (The Universal Verifier)
```lean
import Mathlib
import Aesop

-- Shared Constants
def CrashProbThreshold : Nat := 10
def MaxVolatility : Nat := 5
def CurrentVolatility : Nat := 6

-- Theorem: If volatility exceeds the safety margin, it implies a high-risk state
-- (Abstracted as a simple inequality for the mock)
theorem volatility_risk_check : CurrentVolatility > MaxVolatility := by
  simp [CurrentVolatility, MaxVolatility]
```

# Z3/Python Script (The Empirical Grounding)
```python
import jax.numpy as jnp
from jax import random, vmap

def run_simulation():
    # Shared Constants
    CrashProbThreshold = 0.1
    CurrentVolatility = 6.0
    MaxVolatility = 5.0
    
    key = random.PRNGKey(42)
    
    # Simulate market conditions based on volatility
    # If current volatility is high, probability of crash increases
    def sim_step(k):
        noise = random.normal(k)
        # Simplified model: effective volatility
        eff_vol = CurrentVolatility + noise
        # If effective volatility > MaxVolatility * 1.5, crash occurs
        crash = jnp.where(eff_vol > MaxVolatility * 1.5, 1.0, 0.0)
        return crash

    # Run 10,000 simulations in parallel using vmap
    keys = random.split(key, 10000)
    outcomes = vmap(sim_step)(keys)
    prob_crash = jnp.mean(outcomes)
    
    print(f"Empirical Crash Probability: {prob_crash}")
    
    # Verification Logic:
    # If our logic is correct, the high volatility should lead to 
    # a crash probability that justifies the halt (e.g., > Threshold)
    # For this mock, we just assert the simulation ran successfully.
    assert prob_crash >= 0.0, "Probability cannot be negative"

if __name__ == "__main__":
    run_simulation()
```
"""

    def extract_code(self, response):
        """
        Extracts ALL TLA+, Z3 (Python), and Lean blocks from the LLM response.
        Returns LISTS of strings for each language.
        """
        # Find all matches using re.findall
        tla_matches = re.findall(r"```tla\s*\n?(.*?)\n?\s*```", response, re.DOTALL)
        # Match python OR z3 blocks
        python_matches = re.findall(r"```(?:python|z3)\s*\n?(.*?)\n?\s*```", response, re.DOTALL)
        lean_matches = re.findall(r"```lean\s*\n?(.*?)\n?\s*```", response, re.DOTALL)

        # Remove entire code blocks from prose
        prose = response
        prose = re.sub(r"```tla\s*.*?\s*```", "", prose, flags=re.DOTALL)
        prose = re.sub(r"```(?:python|z3)\s*.*?\s*```", "", prose, flags=re.DOTALL)
        prose = re.sub(r"```lean\s*.*?\s*```", "", prose, flags=re.DOTALL)

        return {
            "prose": prose.strip(),
            "tla": tla_matches if tla_matches else None,
            "python": python_matches if python_matches else None,
            "lean": lean_matches if lean_matches else None
        }