import logging
from .features.url_features import extract_features, validate_url
from .models.predictor import PhishingPredictor
from .heuristics import create_engine as create_heuristic_engine
from .risk.engine import RiskEngine
from .types import DetectionResult

logger = logging.getLogger(__name__)

class PhishingDetector:
    def __init__(self, model_path='models/benchmarks/char_cnn.pkl'):
        self.predictor = PhishingPredictor(model_path)
        self.heuristic_engine = create_heuristic_engine()
        self.risk_engine = RiskEngine()
        
    def analyze(self, url: str) -> dict:
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
            
            # 1. Run Heuristics
            heuristic_findings = self.heuristic_engine.evaluate(url)
            
            # 2. Package Detection Result
            detection = DetectionResult(
                url=url,
                ml_probability=pred_res['confidence'],
                ml_model_name='char_cnn',
                is_calibrated_probability=False,
                heuristic_findings=heuristic_findings
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
