from src.main import FormalReasoningLoop
from pydantic import BaseModel
from typing import Optional, Dict, Any

class VerificationRequest(BaseModel):
    task: str
    model: Optional[str] = "gemini-2.5-flash"
    
class VerificationResponse(BaseModel):
    success: bool
    answer: Optional[str]
    artifacts: Dict[str, Any] = {}

class ProofLoopAPI:
    def __init__(self):
        # We initialize the loop lazily or per request? 
        # The loop has state (library manager). It's better to instantiate per request 
        # or keep a persistent one if we want caching.
        # Given the CLI usage, persistent is fine.
        pass

    def verify(self, request: VerificationRequest) -> VerificationResponse:
        # Instantiate the loop
        loop = FormalReasoningLoop(
            model=request.model,
            verbose=False # Keep logs clean
        )
        
        # Capture stdout? Ideally we refactor main.py to not print, 
        # but for now we trust the return value.
        # The run method returns (success, final_analysis)
        
        success, analysis = loop.run(request.task)
        
        return VerificationResponse(
            success=success,
            answer=analysis,
            artifacts={
                "proof_dir": "See library/ for details" # We could return paths
            }
        )

# Singleton instance
api = ProofLoopAPI()
