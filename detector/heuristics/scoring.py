from enum import Enum
from typing import List, Dict, Any
from urllib.parse import urlparse

class Severity(Enum):
    SAFE = "SAFE"
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

import inspect

class HeuristicEngine:
    def __init__(self):
        self.rules = []

    def register_rule(self, rule_func):
        self.rules.append(rule_func)

    def evaluate(self, url: str, page_signals: dict = None, brand_findings: dict = None) -> List[RuleMatch]:
        try:
            if not url.startswith(('http://', 'https://')):
                url = f"http://{url}"
            parsed = urlparse(url)
        except Exception:
            return []
            
        matches = []
        for rule in self.rules:
            try:
                sig = inspect.signature(rule)
                kwargs = {}
                if 'page_signals' in sig.parameters:
                    kwargs['page_signals'] = page_signals
                if 'brand_findings' in sig.parameters:
                    kwargs['brand_findings'] = brand_findings
                    
                match = rule(url, parsed, **kwargs)
                if match:
                    if isinstance(match, list):
                        matches.extend(match)
                    else:
                        matches.append(match)
            except Exception:
                pass
        return matches
