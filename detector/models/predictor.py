import pickle
import pandas as pd
import logging
import os

logger = logging.getLogger(__name__)

class PhishingPredictor:
    def __init__(self, model_path='models/phishing_detector.pkl'):
        self.model_path = model_path
        self.model = None
        self._load_model()

    def _load_model(self):
        try:
            if os.path.exists(self.model_path):
                with open(self.model_path, 'rb') as f:
                    self.model = pickle.load(f)
            else:
                logger.error(f"Model file not found at {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to load model from {self.model_path}: {e}")

    def predict(self, url: str, features: dict = None) -> dict:
        if not self.model:
            raise RuntimeError("Model is not loaded.")
        
        # Check if the model is CharCNN (expects a list of raw URLs)
        if hasattr(self.model, '_tokenize') or type(self.model).__name__ == 'CharCNNClassifier':
            prediction = int(self.model.predict([url])[0])
            confidence = float(self.model.predict_proba([url])[0][1])
        else:
            if not features:
                raise ValueError("Features dict required for tree-based models.")
            feature_df = pd.DataFrame([features])
            
            # Ensure correct features exist
            expected_features = ['url_length', 'has_ip', 'https', 'num_dots', 'num_slashes', 
                                 'has_at', 'has_dash', 'has_query', 'domain_length', 
                                 'tld_length', 'has_subdomain', 'dns_record']
            
            for col in expected_features:
                if col not in feature_df:
                    feature_df[col] = 0
                    
            if hasattr(self.model, "feature_names_in_"):
                feature_df = feature_df[self.model.feature_names_in_]
                
            prediction = int(self.model.predict(feature_df)[0])
            confidence = float(self.model.predict_proba(feature_df)[0][prediction])
        
        return {
            'prediction': prediction,
            'confidence': confidence
        }
