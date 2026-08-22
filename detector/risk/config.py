from detector.heuristics.scoring import Severity
from detector.types import Action

# Risk Config - All weights and thresholds configurable

# Base ML config
ML_WEIGHT = 100  # The base score multiplier for ML probability (e.g., 0.85 * 100 = 85)

# Heuristic penalty weights
HEURISTIC_PENALTIES = {
    Severity.CRITICAL: 40,
    Severity.HIGH: 25,
    Severity.MEDIUM: 10,
    Severity.LOW: 5
}

# Final Severity and Action Thresholds based on Risk Score (0-100)
RISK_THRESHOLDS = {
    Severity.CRITICAL: {"min_score": 85, "action": Action.BLOCK},
    Severity.HIGH: {"min_score": 70, "action": Action.WARN},
    Severity.MEDIUM: {"min_score": 40, "action": Action.WARN},
    Severity.LOW: {"min_score": 0, "action": Action.ALLOW}
}
