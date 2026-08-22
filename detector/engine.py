import logging
from .features.url_features import extract_features, validate_url
from .models.predictor import PhishingPredictor

logger = logging.getLogger(__name__)

class PhishingDetector:
    def __init__(self, model_path='models/phishing_detector.pkl'):
        self.predictor = PhishingPredictor(model_path)
        
    def analyze(self, url: str) -> dict:
        """
        Analyzes a URL and returns standard prediction output.
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
                
            pred_res = self.predictor.predict(features)
            
            result = 'Phishing' if pred_res['prediction'] == 1 else 'Legitimate'
            
            return {
                'url': url,
                'result': result,
                'confidence': round(pred_res['confidence'] * 100, 2)
            }
        except Exception as e:
            logger.error(f"Error analyzing URL {url}: {e}")
            return {
                'url': url,
                'result': 'Error',
                'confidence': 0.0,
                'error': str(e)
            }
