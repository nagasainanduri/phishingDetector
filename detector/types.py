from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum
from detector.heuristics.scoring import RuleMatch, Severity

class Action(Enum):
    ALLOW = "ALLOW"
    WARN = "WARN"
    BLOCK = "BLOCK"

@dataclass
class DetectionResult:
    raw_url: str
    canonical_url: str
    model_probability: float
    ml_model_name: str
    is_calibrated_probability: bool = False
    canonicalization_findings: Dict[str, Any] = field(default_factory=dict)
    heuristic_findings: List[RuleMatch] = field(default_factory=list)
    future_page_findings: List[Any] = field(default_factory=list)
    future_brand_findings: List[Any] = field(default_factory=list)
    future_threat_intel: List[Any] = field(default_factory=list)
    model_explanation: List[Dict[str, float]] = field(default_factory=list)
    explanation_limitation: Optional[str] = None
    privacy_mode: str = "local_only"
    telemetry: bool = False


@dataclass
class RiskAssessment:
    risk_score: int
    severity: Severity
    recommended_action: Action
    evidence_summary: List[str]
