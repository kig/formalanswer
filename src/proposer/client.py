import re
import os
import subprocess
import json
import time
from google import genai
from dotenv import load_dotenv
from .prompts import SYSTEM_PROMPT, format_user_prompt, RAP_BATTLE_RULES, ADVERSARIAL_COMBAT_RULES, PEER_REVIEW_RULES
from .repair_prompt import REPAIR_PROMPT
from .rap_repair_prompt import RAP_REPAIR_PROMPT

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

# Load environment variables from .env
load_dotenv()

from typing import NamedTuple, Dict, Optional

class Usage(NamedTuple):
    input_tokens: int
    output_tokens: int

class ProposerResponse(NamedTuple):
    content: str
    usage: Usage

class Proposer:
    def __init__(self, backend="gemini", model_name=None, api_key=None, base_url=None):
        self.backend = backend.lower()
        self.model_name = model_name
        self.history = []  # For stateless backends like OpenAI/Ollama/Gemini-CLI
        self.usage_input = 0
        self.usage_output = 0

        if self.backend == "mock":
            print("[PROPOSER] Using MOCK backend.")
            self.model_name = "mock"
            self.client = None
            
        elif self.backend == "gemini":
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
            
        elif self.backend == "gemini-cli":
            print("[PROPOSER] Using GEMINI-CLI backend (Dogfooding).")
            self.model_name = self.model_name or "default"
            self.history = [{"role": "system", "content": SYSTEM_PROMPT}]
            
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

    def get_usage(self) -> Usage:
        return Usage(self.usage_input, self.usage_output)

    def _call_gemini_cli(self, prompt: str) -> ProposerResponse:
        """
        Calls the 'gemini' command line tool and parses JSON output.
        """
        cmd = [
            "gemini", 
            "-p", prompt, 
            "--output-format", "json", 
            "--raw-output", 
            "--accept-raw-output-risk"
        ]
        if self.model_name and self.model_name != "default":
            cmd.extend(["-m", self.model_name])
            
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            output = result.stdout
            json_start = output.find('{')
            if json_start == -1:
                print(f"Gemini-CLI Error: No JSON found in output: {output}")
                return ProposerResponse("Error", Usage(0, 0))
            
            data = json.loads(output[json_start:])
            content = data.get("response", "")
            
            total_in = 0
            total_out = 0
            stats = data.get("stats", {}).get("models", {})
            for m_stats in stats.values():
                tokens = m_stats.get("tokens", {})
                total_in += tokens.get("input", 0)
                total_out += tokens.get("candidates", 0)
            
            usage = Usage(total_in, total_out)
            self.usage_input += total_in
            self.usage_output += total_out
            return ProposerResponse(content, usage)
            
        except Exception as e:
            print(f"Gemini-CLI execution error: {e}")
            return ProposerResponse("Error", Usage(0, 0))

    def propose(self, task, feedback=None, context=None, rap_battle=False, combat=False, peer_review=False, force_mode=None, tier="pro") -> ProposerResponse:
        """
        Calls the LLM API (Stateful).
        """
        if self.backend == "mock":
            self.usage_input += 100
            self.usage_output += 200
            return ProposerResponse(self._get_mock_response(), Usage(100, 200))

        context_prefix = ""
        if rap_battle:
            context_prefix = RAP_BATTLE_RULES
        elif combat:
             context_prefix = ADVERSARIAL_COMBAT_RULES
        elif peer_review:
             context_prefix = PEER_REVIEW_RULES

        tier_prefix = ""
        if tier == "standard":
            tier_prefix = "ASSURANCE TIER: STANDARD. Focus on rapid, direct answers. Minimalist formalization. Be concise.\n"
        elif tier == "enterprise":
            tier_prefix = "ASSURANCE TIER: ENTERPRISE. Maximum rigor required. Exhaustive TLA+ specs and Lean proofs. Model every edge case. High-budget reasoning.\n"
        else:
            tier_prefix = "ASSURANCE TIER: PRO. Balanced rigor and cost. Robust formalization.\n"

        if self.backend == "gemini":
            if feedback:
                repair_tmpl = RAP_REPAIR_PROMPT if rap_battle else REPAIR_PROMPT
                prompt = f"{context_prefix}{tier_prefix}{repair_tmpl.format(feedback=feedback)}"
            else:
                prompt = format_user_prompt(task, context, force_mode=force_mode, context_prefix=context_prefix, tier_prefix=tier_prefix)
            
            try:
                response = self.chat.send_message(prompt)
                usage = Usage(
                    response.usage_metadata.prompt_token_count,
                    response.usage_metadata.candidates_token_count
                )
                self.usage_input += usage.input_tokens
                self.usage_output += usage.output_tokens
                return ProposerResponse(response.text, usage)
            except Exception as e:
                print(f"Gemini API Error: {e}")
                return ProposerResponse(self._get_mock_response(), Usage(0, 0))

        elif self.backend == "gemini-cli":
            if feedback:
                repair_tmpl = RAP_REPAIR_PROMPT if rap_battle else REPAIR_PROMPT
                prompt = f"{context_prefix}{tier_prefix}{repair_tmpl.format(feedback=feedback)}"
                self.history.append({"role": "user", "content": prompt})
            else:
                prompt = format_user_prompt(task, context, force_mode=force_mode, context_prefix=context_prefix, tier_prefix=tier_prefix)
                self.history.append({"role": "user", "content": prompt})
            
            full_prompt = "\n\n".join([f"{m['role'].upper()}: {m['content']}" for m in self.history])
            resp = self._call_gemini_cli(full_prompt)
            self.history.append({"role": "assistant", "content": resp.content})
            return resp

        elif self.backend in ["openai", "ollama"]:
            if feedback:
                repair_tmpl = RAP_REPAIR_PROMPT if rap_battle else REPAIR_PROMPT
                prompt = f"{context_prefix}{tier_prefix}{repair_tmpl.format(feedback=feedback)}"
                self.history.append({"role": "user", "content": prompt})
            else:
                prompt = format_user_prompt(task, context, force_mode=force_mode, context_prefix=context_prefix, tier_prefix=tier_prefix)
                self.history.append({"role": "user", "content": prompt})
            
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=self.history
                )
                content = response.choices[0].message.content
                self.history.append({"role": "assistant", "content": content})
                
                usage = Usage(0, 0)
                if response.usage:
                    usage = Usage(response.usage.prompt_tokens, response.usage.completion_tokens)
                
                self.usage_input += usage.input_tokens
                self.usage_output += usage.output_tokens
                return ProposerResponse(content, usage)
            except Exception as e:
                print(f"{self.backend.upper()} API Error: {e}")
                return ProposerResponse(self._get_mock_response(), Usage(0, 0))

    def explain_trace(self, trace_text, spec_code):
        prompt = (
            "You are a TLA+ Expert Debugger.\n"
            "Below is a TLA+ specification and a counter-example trace produced by TLC.\n"
            "Analyze the trace step-by-step and explain logically WHY the invariant was violated.\n"
            "Keep it concise (3-4 sentences).\n\n"
            f"SPECIFICATION:\n{spec_code[:2000]}...\n\n" 
            f"TRACE:\n{trace_text}\n\n"
            "OUTPUT FORMAT:\n"
            "EXPLANATION: [Your explanation]"
        )
        response = self._call_stateless(prompt)
        if response.content.startswith("EXPLANATION:"):
            return response.content[12:].strip()
        return response.content

    def rap_battle(self, proof_text):
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
        return self._call_stateless(prompt).content

    def rap_judge(self, proof_text, objection):
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
        resp_arg = self._call_stateless(prompt_arg).content
        
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
        resp_roast = self._call_stateless(prompt_roast).content

        arg_score = 3
        roast_score = 3
        arg_comm = ""
        roast_comm = ""

        try:
            m_a = re.search(r"SCORE:\s*(\d+)", resp_arg, re.IGNORECASE)
            if m_a: arg_score = int(m_a.group(1))
            m_ac = re.search(r"COMMENTARY:\s*(.*)", resp_arg, re.IGNORECASE | re.DOTALL)
            if m_ac: arg_comm = m_ac.group(1).strip()

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
        MANDATES a formal counter-proof.
        """
        prompt = (
            "You are a 'Red Team' adversary and Formal Methods Expert. Your goal is to provide a STERN mechanical critique.\n"
            "If the logic is solid, admit it. But if there is any gap, you MUST provide a FORMAL COUNTER-PROOF.\n"
            "1. **Mechanical Failure:** Provide a TLA+ snippet or Lean 4 block that breaks the proponent's invariants.\n"
            "2. **Vacuity Check:** Does the proof pass only because the preconditions are never met? (e.g. proving a property of an empty set).\n"
            "3. **Sim-to-Real Gap:** Demonstrate where the abstract model ignores a critical real-world failure mode.\n"
            "4. **No Hallucinated Constraints:** Do not invent constraints not in the prompt.\n\n"
            f"ARGUMENT TO ATTACK:\n{proof_text}\n\n"
            "OUTPUT FORMAT:\n"
            "OBJECTION: [One clear objection].\n"
            "COUNTER-PROOF: [MANDATORY: Code block (TLA+ or Lean) demonstrating the flaw].\n"
            "SEVERITY: [High/Medium/Low]."
        )
        return self._call_stateless(prompt).content

    def judge(self, proof_text, objection):
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
        resp_arg = self._call_stateless(prompt_arg).content
        
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
        resp_obj = self._call_stateless(prompt_obj).content

        arg_score = 3
        obj_score = 3
        arg_comm = ""
        obj_comm = ""

        try:
            m_a = re.search(r"SCORE:\s*(\d+)", resp_arg, re.IGNORECASE)
            if m_a: arg_score = int(m_a.group(1))
            m_ac = re.search(r"COMMENTARY:\s*(.*)", resp_arg, re.IGNORECASE | re.DOTALL)
            if m_ac: arg_comm = m_ac.group(1).strip()

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
        return self._call_stateless(prompt).content

    def peer_review_judge(self, proof_text, suggestion):
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
        response = self._call_stateless(prompt).content
        try:
            match = re.search(r"(\d+(\.\d+)?)", response)
            if match:
                return float(match.group(1))
            return 0.5
        except:
            return 0.5

    def construct_final_rap(self, raw_history):
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
        return self._call_stateless(prompt).content

    def produce_song_gen_lyrics(self, rap_script):
        prompt = (
            "You are a Lyrics Formatter. Convert the following Rap Battle Script into Tencent SongGeneration format.\n"
            "FORMAT RULES:\n"
            "1. One paragraph per segment, starting with [structure tag] and ending with blank line.\n"
            "2. One sentence per line. No punctuation inside sentences.\n"
            "3. Tags: [verse], [chorus], [bridge]. Avoid intro/outro/inst tags if they have no lyrics.\n"
            "4. Structure tags must be on their own line.\n\n"
            f"INPUT SCRIPT:\n{rap_script}\n\n"
            "OUTPUT ONLY THE FORMATTED LYRICS."
        )
        return self._call_stateless(prompt).content

    def _call_stateless(self, prompt) -> ProposerResponse:
        """
        Helper for stateless calls (Critique/Judge).
        """
        if self.backend == "mock":
            self.usage_input += 10
            self.usage_output += 20
            return ProposerResponse("Mock Response", Usage(10, 20))
        
        if self.backend == "gemini-cli":
            return self._call_gemini_cli(prompt)

        if self.backend == "gemini":
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt
                )
                usage = Usage(
                    response.usage_metadata.prompt_token_count,
                    response.usage_metadata.candidates_token_count
                )
                self.usage_input += usage.input_tokens
                self.usage_output += usage.output_tokens
                return ProposerResponse(response.text, usage)
            except Exception as e:
                print(f"Gemini API Error (Stateless): {e}")
                return ProposerResponse("Error", Usage(0, 0))
        elif self.backend in ["openai", "ollama"]:
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}]
                )
                content = response.choices[0].message.content
                usage = Usage(0, 0)
                if response.usage:
                    usage = Usage(response.usage.prompt_tokens, response.usage.completion_tokens)
                self.usage_input += usage.input_tokens
                self.usage_output += usage.output_tokens
                return ProposerResponse(content, usage)
            except Exception as e:
                print(f"{self.backend.upper()} API Error (Stateless): {e}")
                return ProposerResponse("Error", Usage(0, 0))
        return ProposerResponse("Error", Usage(0, 0))

    def _get_mock_response(self):
        return """
# Mode Selection
[MODE: DISCRETE]

# Critique & Refinement
- **Critique:** Assumption 1
- **Refinement:** Refined answer.

# Rationale & Shared Constants
We model a simple counter.

Shared Invariant: `val <= Threshold`

Shared Constants:
- `Threshold`: 10

# TLA+ Specification (The Safety Inspector)
```tla
---- MODULE temp ----
EXTENDS Naturals, TLC
VARIABLE val
Init == val = 0
Next == val < 10 /\ val' = val + 1
Spec == Init /\ [][Next]_val
Safety == val <= 10
====
```

# Lean 4 Proof (The Universal Verifier)
```lean
import Mathlib
import Aesop

theorem simple_inc (n : Nat) (h : n < 10) : n + 1 <= 10 := by
  omega
```

# Z3/Python Script (The Empirical Grounding)
```python
print("Simulation running...")
assert 1 + 1 == 2
```
"""

    def extract_code(self, response):
        tla_matches = re.findall(r"```tla\s*\n?(.*?)\n?\s*```", response, re.DOTALL)
        python_matches = re.findall(r"```(?:python|z3)\s*\n?(.*?)\n?\s*```", response, re.DOTALL)
        lean_matches = re.findall(r"```lean\s*\n?(.*?)\n?\s*```", response, re.DOTALL)

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