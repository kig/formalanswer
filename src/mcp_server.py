from mcp.server.fastmcp import FastMCP
from src.api import api, VerificationRequest

# Initialize FastMCP Server
mcp = FastMCP("FormalAnswer")

@mcp.tool()
def formal_verify(query: str, model: str = "gemini-2.5-flash") -> str:
    """
    Verifies a complex logical query, system design, or safety constraint using 
    Formal Methods (Lean 4, TLA+, JAX).
    
    Use this for:
    - Designing distributed systems (consensus, locks).
    - Verifying critical safety invariants.
    - Probabilistic risk assessment.
    
    Args:
        query: The natural language description of the system or logic to verify.
        model: The LLM model to use for reasoning (default: gemini-2.5-flash).
    """
    req = VerificationRequest(task=query, model=model)
    result = api.verify(req)
    
    if result.success:
        return f"VERIFICATION SUCCESSFUL

{result.answer}"
    else:
        return "VERIFICATION FAILED. See logs for details."

if __name__ == "__main__":
    mcp.run()
