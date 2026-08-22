from detector.types import RiskAssessment, DetectionResult, Action
from detector.heuristics.scoring import Severity

class PolicyEngine:
    def __init__(self, warning_threshold: int = 60, block_threshold: int = 90):
        self.warning_threshold = warning_threshold
        self.block_threshold = block_threshold

    def evaluate(self, risk: RiskAssessment, detection: DetectionResult) -> Action:
        """
        Determines the final browser action based on configurable thresholds
        and context, rather than raw model probabilities alone.
        """
        score = risk.risk_score
        
        # 1. Hard Block for Confirmed Threat Intel / Brand Impersonation
        # If the risk engine flagged these, it probably output CRITICAL.
        if risk.severity == Severity.CRITICAL:
            return Action.BLOCK
            
        # 2. Check Thresholds
        if score >= self.block_threshold:
            # Only block if we have corroborating heuristic evidence. 
            # If it's pure ML (no heuristics triggered) but somehow scored 90+, downgrade to WARN.
            if len(detection.heuristic_findings) > 0:
                return Action.BLOCK
            return Action.WARN
            
        if score >= self.warning_threshold:
            return Action.WARN
            
        return Action.ALLOW
