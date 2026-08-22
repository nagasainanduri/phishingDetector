from enum import Enum
from typing import List, Dict, Any
from urllib.parse import urlparse

class Severity(Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

class RuleMatch:
    def __init__(self, rule_id: str, severity: Severity, description: str, evidence: Any):
        self.rule_id = rule_id
        self.severity = severity
        self.description = description
        self.evidence = evidence

    def to_dict(self):
        return {
            "rule_id": self.rule_id,
            "severity": self.severity.value,
            "description": self.description,
            "evidence": self.evidence
        }

class HeuristicEngine:
    def __init__(self):
        self.rules = []

    def register_rule(self, rule_func):
        self.rules.append(rule_func)

    def evaluate(self, url: str) -> List[RuleMatch]:
        try:
            if not url.startswith(('http://', 'https://')):
                url = f"http://{url}"
            parsed = urlparse(url)
        except Exception:
            return []
            
        matches = []
        for rule in self.rules:
            try:
                match = rule(url, parsed)
                if match:
                    if isinstance(match, list):
                        matches.extend(match)
                    else:
                        matches.append(match)
            except Exception:
                pass
        return matches
