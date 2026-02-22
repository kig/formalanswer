import os
from typing import Dict, List, Any
from src.verifiers.common import VerificationResult

class Reporter:
    @staticmethod
    def calculate_assurance_score(results: Dict[str, List[VerificationResult]]) -> float:
        """
        Calculates a Formal Assurance Score (FAS) from 0 to 100.
        Weights:
        - Lean 4 (Deductive): 40%
        - TLA+ (Temporal): 40%
        - Python/Z3 (Empirical): 20%
        """
        score = 0.0
        
        weights = {
            "lean": 40.0,
            "tla": 40.0,
            "python": 20.0
        }
        
        for kind, weight in weights.items():
            res_list = results.get(kind, [])
            if not res_list:
                continue
            
            # Average success in this category
            success_count = sum(1 for r in res_list if r.success)
            total_count = len(res_list)
            
            if total_count > 0:
                score += (success_count / total_count) * weight
                
        return round(score, 1)

    @staticmethod
    def generate_report(task: str, analysis: str, results: Dict[str, List[VerificationResult]], usage_summary: str, tier: str) -> str:
        """
        Generates a comprehensive Markdown report.
        """
        fas = Reporter.calculate_assurance_score(results)
        
        report = f"# FormalAnswer Logic Report\\n"
        report += f"**Task:** {task}\\n"
        report += f"**Assurance Tier:** {tier.upper()}\\n"
        report += f"**Formal Assurance Score (FAS):** {fas}/100\\n\\n"
        report += f"---\\n\\n"
        report += f"## 1. Executive Summary & Verified Analysis\\n{analysis}\\n\\n"
        report += f"---\\n\\n"
        report += f"## 2. Verification Metadata\\n"
        report += f"**Usage Summary:** {usage_summary}\\n\\n"
        report += f"### Verifier Status:\\n"
        
        for kind, res_list in results.items():
            report += f"#### {kind.upper()}\\n"
            if not res_list:
                report += "- Not applied.\\n"
            for idx, res in enumerate(res_list):
                status = "✅ PASS" if res.success else "❌ FAIL"
                report += f"- Block {idx+1}: {status} ({res.message})\\n"
        
        report += f"\\n---\\n"
        report += f"*This report was autonomously generated and verified by the FormalAnswer System 2 Governor.*\\n"
        
        return report