REPAIR_PROMPT = """
**URGENT REPAIR NEEDED**
The previous attempt failed formal verification.

**FAILURES:**
{feedback}

**INSTRUCTIONS:**
1. **FOCUS:** Fix ONLY the errors listed above. Do not improve or change working code.
2. **FORMAT:** Output ONLY the corrected code blocks wrapped in markdown (e.g., ```lean ... ```).
3. **NO PROSE:** do not include "# Mode Selection", "# Rationale", or any explanation text. just the code.
4. **NO REPEATS:** Do not output blocks that already passed (unless they need changes to fix dependencies).

**GOAL:**
Produce a valid, compiling proof that resolves the error.
"""
