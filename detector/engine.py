import logging
logger = logging.getLogger(__name__)
from .features.url_features import extract_features
from .models.predictor import PhishingPredictor
from .heuristics import create_engine as create_heuristic_engine
from .risk.engine import RiskEngine
from .brand.detector import BrandDetector
from .threat_intel.aggregator import ThreatIntelAggregator
from .policy.engine import PolicyEngine
from .types import DetectionResult
from urllib.parse import urlparse
import re

def validate_url(url: str) -> bool:
    if not re.match(r'^https?://', url):
        url = 'http://' + url
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except:
        return False

class PhishingDetector:
    def __init__(self):
        self.predictor = PhishingPredictor()
        self.heuristic_engine = create_heuristic_engine()
        self.risk_engine = RiskEngine()
        self.policy_engine = PolicyEngine()
        self.brand_detector = BrandDetector()
        self.threat_intel = ThreatIntelAggregator()
        
    def analyze(self, url: str, page_signals: dict = None, privacy_mode: str = "local_only") -> dict:
        """
        Analyzes a URL and returns standard prediction output along with risk assessment.
        """
        url = url.strip()
        if not validate_url(url):
            return {
                'url': url,
                'result': 'Error',
                'confidence': 0.0,
                'error': 'Invalid URL format'
            }
            
        try:
            features = extract_features(url)
            if features is None:
                return {
                    'url': url,
                    'result': 'Error',
                    'confidence': 0.0,
                    'error': 'Unable to verify this URL'
                }
                
            pred_res = self.predictor.predict(url, features)
            
            # 1. External Checks
            brand_findings = self.brand_detector.analyze(url)
            threat_intel_findings = self.threat_intel.analyze(url, privacy_mode=privacy_mode)
            
            # 2. Run Heuristics
            heuristic_findings = self.heuristic_engine.evaluate(
                url, 
                page_signals=page_signals, 
                brand_findings=brand_findings,
                threat_intel_findings=threat_intel_findings
            )
            
            # 3. Package Detection Result
            detection = DetectionResult(
                url=url,
                ml_probability=pred_res['raw_probability'] if 'raw_probability' in pred_res else pred_res['confidence'],
                ml_model_name=pred_res['model_name'],
                is_calibrated_probability=pred_res['is_calibrated'],
                heuristic_findings=heuristic_findings,
                future_brand_findings=[brand_findings] if brand_findings else [],
                future_threat_intel=threat_intel_findings,
                model_explanation=pred_res.get('model_explanation', []),
                explanation_limitation=pred_res.get('explanation_limitation')
            )
            
            # 4. Evaluate Risk
            risk = self.risk_engine.evaluate(detection)
            
            # 5. Evaluate Policy
            action = self.policy_engine.evaluate(risk, detection)
            
            # Format model explanation for frontend if it exists
            formatted_explanation = []
            if detection.model_explanation:
                for item in detection.model_explanation:
                    formatted_explanation.append(f"{item['feature'].replace('_', ' ').title()} ({round(item['importance'] * 100)}% impact)")

            return {
                'url': url,
                'result': risk.severity.value, # Return severity string for compatibility or custom handling
                'confidence': round(pred_res['confidence'] * 100, 2),
                'risk_score': risk.risk_score,
                'severity': risk.severity.value,
                'action': action.value,
                'reasons': risk.evidence_summary,
                'model_explanation': formatted_explanation,
                'explanation_limitation': detection.explanation_limitation
            }

        except Exception as e:
            logger.error(f"Error analyzing URL {url}: {e}")
            return {
                'url': url,
                'result': 'Error',
                'confidence': 0.0,
                'error': str(e)
            }
