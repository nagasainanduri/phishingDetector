from detector.types import DetectionResult, RiskAssessment, Action
from detector.heuristics.scoring import Severity
from . import config

class RiskEngine:
    def __init__(self, custom_config=None):
        self.config = custom_config or config

    def evaluate(self, detection: DetectionResult) -> RiskAssessment:
        evidence = []
        
        # 1. Base ML Score
        base_score = int(detection.model_probability * self.config.ML_WEIGHT)
        
        calibrated_str = "Calibrated" if detection.is_calibrated_probability else "Not calibrated"
        evidence.append(f"ML Model ({detection.ml_model_name}) raw probability: {detection.model_probability:.2f} ({calibrated_str}) -> Base score: {base_score}")

        # 2. Add Heuristic Penalties
        heuristic_penalty = 0
        for match in detection.heuristic_findings:
            penalty = self.config.HEURISTIC_PENALTIES.get(match.severity, 0)
            heuristic_penalty += penalty
            evidence.append(f"Heuristic [{match.severity.name}] {match.rule_id}: {match.description} (+{penalty})")

        # 3. Calculate total score
        total_score = base_score + heuristic_penalty
        total_score = max(0, min(100, total_score)) # Clamp 0-100
        
        if heuristic_penalty > 0:
            evidence.append(f"Total risk score adjusted to {total_score}")

        # 4. Determine final Severity and Action
        final_severity = Severity.LOW
        final_action = Action.ALLOW
        
        # Sort thresholds descending
        sorted_thresholds = sorted(self.config.RISK_THRESHOLDS.items(), key=lambda x: x[1]['min_score'], reverse=True)
        
        for sev, thres in sorted_thresholds:
            if total_score >= thres['min_score']:
                final_severity = sev
                final_action = thres['action']
                break
                
        return RiskAssessment(
            risk_score=total_score,
            severity=final_severity,
            recommended_action=final_action,
            evidence_summary=evidence
        )
