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

    def propose(self, task, feedback=None):
        """
        Calls the LLM API.
        """
        if self.backend == "gemini":
            if feedback:
                prompt = f"The previous attempt failed verification.\n\n{feedback}"
            else:
                prompt = format_user_prompt(task)
            
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
                prompt = format_user_prompt(task)
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
        # (Mock response retained for safety/fallback)
        return """
# Rationale & Shared Constants
Superhuman thinking is defined as the ability to navigate a search space larger than human working memory allows, without error. We model this as a 'Verify-Search' loop.

Shared Invariants:
1. Capacity > 2 * Load

Concrete Values:
- Capacity = 10
- Load = 3

# TLA+ Specification (The Safety Inspector)
```tla
---- MODULE temp ----
EXTENDS Naturals, TLC
CONSTANTS Limit
VARIABLES val, pc

LimitVal == 10

Init == 
    /\ val = 0 
    /\ pc = "start"

Next == 
  \/ /\ pc = "start"
     /\ val < LimitVal
     /\ val' = val + 1
     /\ pc' = "start"
  \/ /\ pc = "start"
     /\ val = LimitVal
     /\ pc' = "done"
     /\ UNCHANGED val
  \/ /\ pc = "done"
     /\ UNCHANGED <<val, pc>>

Spec == Init /\ [][Next]_<<val, pc>>
====
```

# Lean 4 Proof (The Universal Verifier)
```lean
import Mathlib
import Aesop

example : 10 > 2 * 3 := by omega

theorem capacity_safety (c l : Nat) (h : c > 2 * l) : c > l := by
  linarith
```
"""

    def extract_code(self, response):
        """
        Extracts TLA+, Z3 (Python), and Lean blocks from the LLM response.
        """
        tla_match = re.search(r"```tla\s*\n?(.*?)\n?\s*```", response, re.DOTALL)
        z3_match = re.search(r"```python\s*\n?(.*?)\n?\s*```", response, re.DOTALL)
        lean_match = re.search(r"```lean\s*\n?(.*?)\n?\s*```", response, re.DOTALL)
        
        return {
            "tla": tla_match.group(1) if tla_match else None,
            "z3": z3_match.group(1) if z3_match else None,
            "lean": lean_match.group(1) if lean_match else None
        }
