import json
import os
from datetime import datetime
from typing import Dict, Any

USAGE_LOG = "usage_meter.json"

# Approximate costs per 1M tokens (2026 estimates)
COST_TABLE = {
    "gemini-2.5-flash": {"input": 0.10, "output": 0.30},
    "gemini-3-pro": {"input": 1.50, "output": 4.50},
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "o1": {"input": 15.00, "output": 60.00},
    "default": {"input": 1.00, "output": 3.00}
}

class Meter:
    def __init__(self):
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.estimated_cost = 0.0
        self.session_start = datetime.now()

    def record_usage(self, model: str, input_tokens: int, output_tokens: int):
        """
        Records token usage and estimates cost.
        """
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        
        rates = COST_TABLE.get(model, COST_TABLE["default"])
        session_cost = (input_tokens / 1_000_000 * rates["input"]) + (output_tokens / 1_000_000 * rates["output"])
        self.estimated_cost += session_cost

    def persist(self, task_name: str):
        """
        Appends session usage to a persistent JSON log.
        """
        entry = {
            "timestamp": datetime.now().isoformat(),
            "task": task_name,
            "input_tokens": self.total_input_tokens,
            "output_tokens": self.total_output_tokens,
            "estimated_cost_usd": round(self.estimated_cost, 6)
        }
        
        history = []
        if os.path.exists(USAGE_LOG):
            try:
                with open(USAGE_LOG, "r") as f:
                    history = json.load(f)
            except:
                pass
        
        history.append(entry)
        with open(USAGE_LOG, "w") as f:
            json.dump(history, f, indent=2)

    def get_session_summary(self) -> str:
        return f"Tokens: {self.total_input_tokens} In / {self.total_output_tokens} Out | Est. Cost: ${self.estimated_cost:.4f}"
