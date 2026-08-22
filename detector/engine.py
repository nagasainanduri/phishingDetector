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
from .canonicalization import URLCanonicalizer
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
        self.canonicalizer = URLCanonicalizer()
        
    def analyze(self, url: str, page_signals: dict = None, privacy_mode: str = "local_only") -> dict:
        """
        Analyzes a URL and returns standard prediction output along with risk assessment.
        Unexpected programming or infrastructure failures will propagate to trigger appropriate 5xx responses.
        """
        url = url.strip()
        if not validate_url(url):
            return {
                'url': url,
                'result': 'Error',
                'confidence': 0.0,
                'error': 'Invalid URL format'
            }

        # 1. Canonicalization
        canonical_result = self.canonicalizer.canonicalize(url)
        c_url = canonical_result.canonical_url

        features = extract_features(c_url)
        if features is None:
            return {
                'url': url,
                'result': 'Error',
                'confidence': 0.0,
                'error': 'Unable to extract features from this URL'
            }
            
        pred_res = self.predictor.predict(c_url, features)
        
        # 2. External Checks (using canonical URL)
        brand_findings = self.brand_detector.analyze(c_url)
        threat_intel_findings = self.threat_intel.analyze(c_url, privacy_mode=privacy_mode)
        
        # 3. Run Heuristics
        heuristic_findings = self.heuristic_engine.evaluate(
            c_url, 
            page_signals=page_signals, 
            brand_findings=brand_findings,
            threat_intel_findings=threat_intel_findings
        )
        
        # 4. Package Detection Result
        canonicalization_findings = {
            'transformations': canonical_result.transformations,
            'encoding_depth': canonical_result.encoding_depth,
            'suspicious_encoding': canonical_result.suspicious_encoding
        }

        detection = DetectionResult(
            raw_url=url,
            canonical_url=c_url,
            model_probability=pred_res['raw_probability'] if 'raw_probability' in pred_res else pred_res['confidence'],
            ml_model_name=pred_res['model_name'],
            is_calibrated_probability=pred_res['is_calibrated'],
            canonicalization_findings=canonicalization_findings,
            heuristic_findings=heuristic_findings,
            future_brand_findings=[brand_findings] if brand_findings else [],
            future_threat_intel=threat_intel_findings,
            model_explanation=pred_res.get('model_explanation', []),
            explanation_limitation=pred_res.get('explanation_limitation')
        )
        
        # 5. Evaluate Risk
        risk = self.risk_engine.evaluate(detection)
        
        # 6. Evaluate Policy
        action = self.policy_engine.evaluate(risk, detection)
        
        # Format model explanation for frontend if it exists
        formatted_explanation = []
        if detection.model_explanation:
            for item in detection.model_explanation:
                # Clarify language as per Stage 19: "contributed to" not "caused"
                formatted_explanation.append(f"{item['feature'].replace('_', ' ').title()} contributed {round(item['importance'] * 100)}% to the model's prediction")

        return {
            'url': url, # Raw URL
            'canonical_url': c_url,
            'result': action.value,
            'model_probability': round(detection.model_probability * 100, 2),
            'confidence': round(pred_res['confidence'] * 100, 2), # Using model's raw confidence score for system confidence
            'risk_score': risk.risk_score,
            'severity': risk.severity.value,
            'action': action.value,
            'reasons': risk.evidence_summary,
            'model_explanation': formatted_explanation,
            'explanation_limitation': detection.explanation_limitation
        }
