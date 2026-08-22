import unittest
from detector.policy.engine import PolicyEngine
from detector.types import RiskAssessment, DetectionResult, Action
from detector.heuristics.scoring import Severity, RuleMatch

class TestPolicyEngine(unittest.TestCase):
    def setUp(self):
        self.policy = PolicyEngine(warning_threshold=60, block_threshold=90)

    def test_allow(self):
        risk = RiskAssessment(risk_score=10, severity=Severity.LOW, evidence_summary=[], recommended_action=Action.ALLOW)
        detection = DetectionResult(url="http://good.com", ml_probability=0.1, ml_model_name="test", is_calibrated_probability=False, heuristic_findings=[])
        
        action = self.policy.evaluate(risk, detection)
        self.assertEqual(action, Action.ALLOW)

    def test_warn_on_ml_only(self):
        # Even if the score is artificially inflated by ML to >90, if there are NO heuristic findings, we downgrade to WARN.
        risk = RiskAssessment(risk_score=95, severity=Severity.HIGH, evidence_summary=[], recommended_action=Action.BLOCK)
        detection = DetectionResult(url="http://bad-ml-only.com", ml_probability=0.99, ml_model_name="test", is_calibrated_probability=False, heuristic_findings=[])
        
        action = self.policy.evaluate(risk, detection)
        self.assertEqual(action, Action.WARN)

    def test_block_with_heuristics(self):
        # Score is >90 AND we have a heuristic match, so it blocks.
        risk = RiskAssessment(risk_score=95, severity=Severity.HIGH, evidence_summary=[], recommended_action=Action.BLOCK)
        match = RuleMatch(rule_id="TST_01", severity=Severity.MEDIUM, description="Test", evidence={})
        detection = DetectionResult(url="http://bad.com", ml_probability=0.95, ml_model_name="test", is_calibrated_probability=False, heuristic_findings=[match])
        
        action = self.policy.evaluate(risk, detection)
        self.assertEqual(action, Action.BLOCK)

    def test_critical_severity_hard_block(self):
        # If threat intel triggered a CRITICAL severity, it overrides thresholds and hard blocks.
        risk = RiskAssessment(risk_score=85, severity=Severity.CRITICAL, evidence_summary=[], recommended_action=Action.BLOCK)
        detection = DetectionResult(url="http://threat.com", ml_probability=0.5, ml_model_name="test", is_calibrated_probability=False, heuristic_findings=[])
        
        action = self.policy.evaluate(risk, detection)
        self.assertEqual(action, Action.BLOCK)

    def test_warn_threshold(self):
        # Score between 60 and 90 -> WARN
        risk = RiskAssessment(risk_score=75, severity=Severity.MEDIUM, evidence_summary=[], recommended_action=Action.ALLOW)
        detection = DetectionResult(url="http://sus.com", ml_probability=0.75, ml_model_name="test", is_calibrated_probability=False, heuristic_findings=[])
        
        action = self.policy.evaluate(risk, detection)
        self.assertEqual(action, Action.WARN)

if __name__ == '__main__':
    unittest.main()
