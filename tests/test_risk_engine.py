import unittest
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from detector.types import DetectionResult, Action
from detector.heuristics.scoring import RuleMatch, Severity
from detector.risk.engine import RiskEngine

class TestRiskEngine(unittest.TestCase):
    def setUp(self):
        self.engine = RiskEngine()

    def test_normal_url(self):
        result = DetectionResult(
            url="http://google.com",
            ml_probability=0.05,
            ml_model_name="cnn"
        )
        assessment = self.engine.evaluate(result)
        self.assertEqual(assessment.severity, Severity.LOW)
        self.assertEqual(assessment.recommended_action, Action.ALLOW)
        self.assertTrue(assessment.risk_score < 40)

    def test_suspicious_url(self):
        result = DetectionResult(
            url="http://192.168.1.1",
            ml_probability=0.95,
            ml_model_name="cnn",
            heuristic_findings=[
                RuleMatch("URL_001", Severity.HIGH, "IP Address", None)
            ]
        )
        assessment = self.engine.evaluate(result)
        self.assertEqual(assessment.severity, Severity.CRITICAL)
        self.assertEqual(assessment.recommended_action, Action.BLOCK)
        self.assertEqual(assessment.risk_score, 100) # Clamped at 100

    def test_conflicting_signals(self):
        # ML misses it, but high heuristics catch it
        result = DetectionResult(
            url="http://paypal.com@attacker.com",
            ml_probability=0.10, # ML says 10%
            ml_model_name="cnn",
            heuristic_findings=[
                RuleMatch("URL_003", Severity.HIGH, "Credentials", None), # +25
                RuleMatch("DOM_004", Severity.HIGH, "Typosquat", None) # +25
            ]
        )
        assessment = self.engine.evaluate(result)
        # Base 10 + 25 + 25 = 60
        self.assertEqual(assessment.risk_score, 60)
        self.assertEqual(assessment.severity, Severity.MEDIUM)
        self.assertEqual(assessment.recommended_action, Action.WARN)
        
    def test_missing_signals(self):
        # Only ML, no heuristics run
        result = DetectionResult(
            url="http://phishing.com",
            ml_probability=0.80,
            ml_model_name="cnn",
            heuristic_findings=[]
        )
        assessment = self.engine.evaluate(result)
        self.assertEqual(assessment.risk_score, 80)
        self.assertEqual(assessment.severity, Severity.HIGH)
        self.assertEqual(assessment.recommended_action, Action.WARN)

if __name__ == '__main__':
    unittest.main()
