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

    def propose(self, task, feedback=None, context=None, rap_battle=False, combat=False, peer_review=False):
        """
        Calls the LLM API (Stateful).
        """
        context_prefix = ""
        if rap_battle:
            context_prefix = (
                "CONTEXT: You are in a formal Logic Rap Battle. \n"
                "**MANDATORY STYLE RULES:**\n"
                "1. **ALL PROSE MUST BE IN RAP VERSE.** (Mode Selection, Critique, Rationale).\n"
                "2. **DO NOT RHYME INSIDE CODE BLOCKS.** The TLA+, Lean, and Python code must be valid, compilable code.\n"
                "3. **End your response** with a 4-8 bar rap verse summarizing your answer (after the code blocks).\n"
                "4. Maintain rigorous logic, but express it through flow and rhyme.\n"
                "5. **NO APOLOGETICS:** The opponent cannot be placated. Attack is the best defense. Only respond to the rhymes.\n"
                "6. **TRUTH IS THE WEAPON:** Your claims must be provably true. That is your means to win.\n\n"
            )
        elif combat:
             context_prefix = (
                "CONTEXT: You are in ADVERSARIAL COMBAT MODE.\n"
                "Your output will be judged against a ruthless 'Red Team' review.\n"
                "**RULES OF ENGAGEMENT:**\n"
                "1. **NO APOLOGETICS:** Do not respond to the reviewer or critique. Do not try to be polite or placate the reviewer.\n"
                "2. **ATTACK IS THE BEST DEFENSE:** Pre-emptively refute potential objections.\n"
                "3. **TRUTH IS THE BEST ATTACK:** Be rigorously correct. Any weakness will be exploited.\n\n"
             )
        elif peer_review:
             context_prefix = (
                "CONTEXT: You are in PEER REVIEW MODE.\n"
                "Your output will be reviewed by a helpful but rigorous colleague.\n"
                "**RULES OF ENGAGEMENT:**\n"
                "1. **COLLABORATIVE RIGOR:** Be detailed and clear. The goal is to build a rock-solid proof together.\n"
                "2. **OPENNESS:** Be prepared to refine your logic based on constructive feedback.\n"
                "3. **PRECISION:** Ensure all formal proofs are complete and well-documented.\n\n"
             )

        if self.backend == "gemini":
            if feedback:
                persona = ""
                prompt = (
                    "The previous attempt failed verification.\n"
                    "INSTRUCTIONS:\n"
                    "1. **REWRITE:** Your response must contain ONLY the corrected argument and proofs. Nothing else.\n"
                    "2. **PRIORITY:** The most important part is the **corrected formal proofs**.\n"
                    "3. Rewrite only the failed proofs.\n"
                    "FEEDBACK:\n"
                    f"{feedback}\n\n"
                    f"{context_prefix}"
                )
            else:
                prompt = f"{context_prefix}{format_user_prompt(task, context)}"
            
            try:
                response = self.chat.send_message(prompt)
                return response.text
            except Exception as e:
                print(f"Gemini API Error: {e}")
                return self._get_mock_response()

        elif self.backend in ["openai", "ollama"]:
            if feedback:
                prompt = (
                    f"{context_prefix}"
                    "The previous attempt failed verification or requires improvement based on review.\n"
                    f"{feedback_header}\n"
                    f"{feedback}\n\n"
                    "INSTRUCTIONS:\n"
                    "1. **IGNORE THE REVIEWER:** Do not talk to, argue with, or acknowledge the reviewer/judge. Do not say 'The reviewer is right' or 'I will fix this'.\n"
                    "2. **ONLY TRUTH:** Your response must contain ONLY the corrected argument and proofs. Nothing else.\n"
                    "3. **MAINTAIN PERSONA:** If in Rap Battle, start immediately with the verse. If in Logic Mode, start immediately with Mode Selection.\n"
                    "4. **PRIORITY:** The most important part is the **corrected formal proofs**.\n"
                    "5. **Format:** Go straight to the standard 5-section format. NO PREAMBLE. NO APOLOGIES. NO META-COMMENTARY.\n"
                    "6. Regenerate the entire response (Mode, Critique, Rationale, Proofs)."
                )
                self.history.append({"role": "user", "content": prompt})
            else:
                prompt = f"{context_prefix}{format_user_prompt(task, context)}"
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

    def rap_battle(self, proof_text):
        """
        Rap Battle Review: Stateless call to find flaws with style.
        """
        prompt = (
            "You are a Legendary Logic Battle Rapper. Your goal is to destroy the following argument with facts and logic.\n"
            "Drop a short verse (4-8 bars) exposing the flaws, then list the technical objections.\n"
            "Even if the proofs look correct, find the weak link.\n"
            "1. Hidden assumptions.\n"
            "2. Hallucinations.\n"
            "3. Sim-to-Real gaps.\n"
            "4. **No Hallucinated Constraints:** Do not invent constraints that are not in the prompt.\n\n"
            f"ARGUMENT TO ROAST:\n{proof_text}\n\n"
            "OUTPUT FORMAT:\n"
            "ROAST (VERSE): [Rhyming verse exposing the flaw].\n"
            "OBJECTION: [Technical explanation of the flaw].\n"
            "SEVERITY: [High/Medium/Low]."
        )
        return self._call_stateless(prompt)

    def rap_judge(self, proof_text, objection):
        """
        Scoring: Evaluates who won the bar using two independent calls.
        """
        # Call 1: Rate the Argument
        prompt_arg = (
            "You are a Logic Rap Battle Judge. Rate the ARGUMENT below.\n"
            f"ARGUMENT: {proof_text[:2000]}...\n\n"
            "INSTRUCTIONS:\n"
            "1. Rate the ARGUMENT (1-5) on logic, flow, and rigor.\n"
            "2. Ignore any potential counter-arguments for now. Focus on the internal strength.\n"
            "OUTPUT FORMAT:\n"
            "SCORE: [1-5]\n"
            "COMMENTARY: [A short 4-bar rap verse explaining your rating]\n"
        )
        resp_arg = self._call_stateless(prompt_arg)
        
        # Call 2: Rate the Roast
        prompt_roast = (
            "You are a Logic Rap Battle Judge. Rate the ROAST below.\n"
            f"CONTEXT (THE ARGUMENT BEING ROASTED): {proof_text[:2000]}...\n"
            f"ROAST: {objection}\n\n"
            "INSTRUCTIONS:\n"
            "1. Rate the ROAST (1-5) on effectiveness, flaw exposure, and style.\n"
            "2. **Guard against Roast Bias:** Do not score the Roast higher just because it is a critique. Critique is easier than construction. Require substantial logic for a high score.\n"
            "3. **Hallucination Check:** If Roast invents constraints not in the context, give it a 1.\n"
            "4. **Ignore 'Severity' claims.** Judge effectiveness yourself.\n"
            "OUTPUT FORMAT:\n"
            "SCORE: [1-5]\n"
            "COMMENTARY: [A short 4-bar rap verse explaining your rating]\n"
        )
        resp_roast = self._call_stateless(prompt_roast)

        arg_score = 3
        roast_score = 3
        arg_comm = ""
        roast_comm = ""

        try:
            # Parse Argument Score
            m_a = re.search(r"SCORE:\s*(\d+)", resp_arg, re.IGNORECASE)
            if m_a: arg_score = int(m_a.group(1))
            m_ac = re.search(r"COMMENTARY:\s*(.*)", resp_arg, re.IGNORECASE | re.DOTALL)
            if m_ac: arg_comm = m_ac.group(1).strip()

            # Parse Roast Score
            m_r = re.search(r"SCORE:\s*(\d+)", resp_roast, re.IGNORECASE)
            if m_r: roast_score = int(m_r.group(1))
            m_rc = re.search(r"COMMENTARY:\s*(.*)", resp_roast, re.IGNORECASE | re.DOTALL)
            if m_rc: roast_comm = m_rc.group(1).strip()

            final_score = 1.0 if arg_score >= roast_score else 0.0
            full_commentary = f"Arg Score: {arg_score} ({arg_comm})\nRoast Score: {roast_score} ({roast_comm})"
            return final_score, full_commentary
        except:
             return 0.5, "Error parsing independent judge responses."

    def critique(self, proof_text):
        """
        Adversarial Review: Stateless call to find flaws.
        """
        prompt = (
            "You are a 'Red Team' adversary. Your goal is to provide a STRICT BUT FAIR critique.\n"
            "Do NOT invent flaws if the argument is solid. However, you must rigorously check for:\n"
            "1. Hidden assumptions or tautologies.\n"
            "2. False premises in the 'Rationale'.\n"
            "3. Gap between the formal model and reality (the 'Sim-to-Real' gap).\n"
            "4. **No Hallucinated Constraints:** Do not invent constraints (e.g. budget, room size, deadline) that are not in the prompt. Stick to the provided information.\n"
            "You MAY provide a formal counter-proof (TLA+, Lean, Python) if it helps demonstrate the flaw.\n\n"
            f"ARGUMENT TO ATTACK:\n{proof_text}\n\n"
            "OUTPUT FORMAT:\n"
            "OBJECTION: [One clear, devastating objection].\n"
            "COUNTER-PROOF (Optional): [Code block if needed].\n"
            "SEVERITY: [High/Medium/Low]."
        )
        return self._call_stateless(prompt)

    def judge(self, proof_text, objection):
        """
        Scoring: Evaluates if the proof survives the objection using two independent calls.
        """
        # Call 1: Rate the Argument
        prompt_arg = (
            "You are an impartial Judge. Rate the ARGUMENT below.\n"
            f"ARGUMENT: {proof_text[:2000]}...\n\n"
            "INSTRUCTIONS:\n"
            "1. Rate the ARGUMENT (1-5) on robustness and correctness.\n"
            "2. Ignore any potential counter-arguments for now. Focus on the internal strength.\n"
            "OUTPUT FORMAT:\n"
            "SCORE: [1-5]\n"
            "COMMENTARY: [Brief assessment]\n"
        )
        resp_arg = self._call_stateless(prompt_arg)
        
        # Call 2: Rate the Objection
        prompt_obj = (
            "You are an impartial Judge. Rate the OBJECTION below.\n"
            f"CONTEXT (THE ARGUMENT BEING CRITIQUED): {proof_text[:2000]}...\n"
            f"OBJECTION: {objection}\n\n"
            "INSTRUCTIONS:\n"
            "1. Rate the OBJECTION (1-5) on validity and severity.\n"
            "2. **Ignore 'Severity' claims.** Judge validity yourself.\n"
            "3. **Hallucination Check:** If Objection invents constraints not in the context, give it a 1.\n"
            "OUTPUT FORMAT:\n"
            "SCORE: [1-5]\n"
            "COMMENTARY: [Brief assessment]\n"
        )
        resp_obj = self._call_stateless(prompt_obj)

        arg_score = 3
        obj_score = 3
        arg_comm = ""
        obj_comm = ""

        try:
            # Parse Argument Score
            m_a = re.search(r"SCORE:\s*(\d+)", resp_arg, re.IGNORECASE)
            if m_a: arg_score = int(m_a.group(1))
            m_ac = re.search(r"COMMENTARY:\s*(.*)", resp_arg, re.IGNORECASE | re.DOTALL)
            if m_ac: arg_comm = m_ac.group(1).strip()

            # Parse Objection Score
            m_o = re.search(r"SCORE:\s*(\d+)", resp_obj, re.IGNORECASE)
            if m_o: obj_score = int(m_o.group(1))
            m_oc = re.search(r"COMMENTARY:\s*(.*)", resp_obj, re.IGNORECASE | re.DOTALL)
            if m_oc: obj_comm = m_oc.group(1).strip()

            final_score = 1.0 if arg_score >= obj_score else 0.0
            full_commentary = f"Arg Score: {arg_score} ({arg_comm})\nObjection Score: {obj_score} ({obj_comm})"
            return final_score, full_commentary
        except:
             return 0.5, "Error parsing independent judge responses."

    def peer_review(self, proof_text):
        """
        Peer Review: Stateless call to find improvements politely.
        """
        prompt = (
            "You are a 'Helpful Colleague' and expert reviewer. Your goal is to help improve the following argument.\n"
            "Please review the text for:\n"
            "1. Clarity and flow.\n"
            "2. Missing edge cases or slight logical gaps.\n"
            "3. Potential for stronger formal guarantees.\n"
            "You MAY provide a formal example (TLA+, Lean) to clarify your suggestion.\n\n"
            f"ARGUMENT TO REVIEW:\n{proof_text}\n\n"
            "OUTPUT FORMAT:\n"
            "SUGGESTION: [Constructive feedback and specific suggestions for improvement].\n"
            "FORMAL_HINT (Optional): [Code block if needed].\n"
            "TONE: Polite, constructive, and collaborative."
        )
        return self._call_stateless(prompt)

    def peer_review_judge(self, proof_text, suggestion):
        """
        Scoring: Evaluates if the suggestion is critical for the argument's validity.
        """
        prompt = (
            "You are an impartial Editor.\n"
            f"ARGUMENT: {proof_text[:2000]}...\n"
            f"SUGGESTION: {suggestion}\n\n"
            "Does this suggestion reveal a significant flaw or just a minor improvement?\n"
            "CRITICAL SCORING RULES:\n"
            "1. If the suggestion identifies a missing critical component (like a missing algorithm, physical constraint, or logical gap), the score should be low (< 0.7) to trigger a fix.\n"
            "2. If the suggestion is just style/clarity, score > 0.8.\n"
            "OUTPUT: A score from 0.0 (Critical Gap) to 1.0 (Solid/Minor Polish needed).\n"
            "Just output the number."
        )
        response = self._call_stateless(prompt)
        try:
            # Extract float
            match = re.search(r"(\d+(\.\d+)?)", response)
            if match:
                return float(match.group(1))
            return 0.5
        except:
            return 0.5

    def construct_final_rap(self, raw_history):
        """
        Compiles the entire history of the battle into a final track.
        """
        prompt = (
            "You are a Legendary Logic Battle Rapper and Producer.\n"
            "Below is the raw history of a Formal Logic Rap Battle (Verses, Roasts, and Rebuttals).\n"
            "Your goal is to compile this into a single, coherent 3-minute Rap Battle Script.\n"
            "FORMAT:\n"
            "- Line-based lyrics.\n"
            "- [VERSE] headers on their own line.\n"
            "- [CHORUS] if applicable.\n"
            "- Include speaker labels (PROPOSER vs OPPONENT).\n"
            "- Smooth out transitions but keep the original logical content.\n\n"
            f"RAW BATTLE HISTORY:\n{raw_history}\n\n"
            "OUTPUT THE FINAL SCRIPT ONLY."
        )
        return self._call_stateless(prompt)

    def _call_stateless(self, prompt):
        """
        Helper for stateless calls (Critique/Judge).
        """
        if self.backend == "gemini":
            try:
                # Use client.models.generate_content for stateless
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt
                )
                return response.text
            except Exception as e:
                print(f"Gemini API Error (Stateless): {e}")
                return "Error"
        elif self.backend in ["openai", "ollama"]:
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}]
                )
                return response.choices[0].message.content
            except Exception as e:
                print(f"{self.backend.upper()} API Error (Stateless): {e}")
                return "Error"
        return "Error"

    def _get_mock_response(self):
        # A robust mock response demonstrating the Probabilistic/Predictive architecture
        return """
# Mode Selection
[MODE: PROBABILISTIC]

# Critique & Refinement
- **Critique:** Initial assumption X is too strong.
- **Refinement:** Relax assumption X to Y.

# Rationale & Shared Constants
We model a generic system with a threshold.

Shared Invariant: `Metric > Threshold -> State = DONE`

Shared Constants:
- `Threshold`: 10
- `CurrentValue`: 5

# TLA+ Specification (The Safety Inspector)
```tla
---- MODULE GenericSystem ----
EXTENDS Naturals, TLC
CONSTANTS Threshold
VARIABLES val
Init == val = 0
Next == val' = val + 1
Spec == Init /\ [][Next]_<<val>>
====
```

# Lean 4 Proof (The Universal Verifier)
```lean
import Mathlib
import Aesop

theorem simple_math (x : Nat) : x + 0 = x := by simp
```

# Z3/Python Script (The Empirical Grounding)
```python
print("Simulation running...")
assert True
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
