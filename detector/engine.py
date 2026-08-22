import logging
logger = logging.getLogger(__name__)
from .features.url_features import extract_features
from .models.predictor import PhishingPredictor
from .heuristics import create_engine as create_heuristic_engine
from .risk.engine import RiskEngine
from .brand.detector import BrandDetector
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
        self.brand_detector = BrandDetector()
        
    def analyze(self, url: str, page_signals: dict = None) -> dict:
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
            
            # 1. Run Brand Detection
            brand_findings = self.brand_detector.analyze(url)
            
            # 2. Run Heuristics
            heuristic_findings = self.heuristic_engine.evaluate(url, page_signals=page_signals, brand_findings=brand_findings)
            
            # 3. Package Detection Result
            detection = DetectionResult(
                url=url,
                ml_probability=pred_res['raw_probability'],
                ml_model_name=pred_res['model_name'],
                is_calibrated_probability=pred_res['is_calibrated'],
                heuristic_findings=heuristic_findings,
                future_brand_findings=[brand_findings] if brand_findings else []
            )
            
            # 3. Evaluate Risk
            risk = self.risk_engine.evaluate(detection)
            
            return {
                'url': url,
                'result': risk.severity.value, # Return severity string for compatibility or custom handling
                'confidence': round(pred_res['confidence'] * 100, 2),
                'risk_score': risk.risk_score,
                'severity': risk.severity.value,
                'recommended_action': risk.recommended_action.value,
                'reasons': risk.evidence_summary
            }

        except Exception as e:
            logger.error(f"Error analyzing URL {url}: {e}")
            return {
                'url': url,
                'result': 'Error',
                'confidence': 0.0,
                'error': str(e)
            }
