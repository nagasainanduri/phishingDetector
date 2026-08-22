import pickle
import pandas as pd
import logging
import os
import warnings

try:
    from sklearn.exceptions import InconsistentVersionWarning
    warnings.simplefilter("ignore", InconsistentVersionWarning)
except ImportError:
    pass

logger = logging.getLogger(__name__)

class PhishingPredictor:
    def __init__(self, model_path='models/phishing_detector.pkl'):
        self.model_path = model_path
        self.model = None
        self.explainer = None
        self._load_model()

    def _load_model(self):
        try:
            if os.path.exists(self.model_path):
                with open(self.model_path, 'rb') as f:
                    self.model = pickle.load(f)
                    
                # Try to initialize SHAP for Tree Models
                # For CharCNN, it's not practical to attribute character-level features for human readability
                model_type = type(self.model).__name__
                if model_type in ['RandomForestClassifier', 'DecisionTreeClassifier', 'GradientBoostingClassifier']:
                    try:
                        import shap
                        self.explainer = shap.TreeExplainer(self.model)
                        logger.info("Initialized SHAP TreeExplainer.")
                    except ImportError:
                        logger.warning("SHAP not installed. Model explainability disabled.")
            else:
                logger.error(f"Model file not found at {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to load model from {self.model_path}: {e}")

    def predict(self, url: str, features: dict = None) -> dict:
        if not self.model:
            raise RuntimeError("Model is not loaded.")
        
        explanation = []
        explanation_limitation = None
        
        # Check if the model is CharCNN
        if hasattr(self.model, '_tokenize') or type(self.model).__name__ == 'CharCNNClassifier':
            prediction = int(self.model.predict([url])[0])
            confidence = float(self.model.predict_proba([url])[0][1])
            explanation_limitation = "Feature attribution is not supported for sequence-based Deep Learning models (CharCNN)."
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
            
            # Explain with SHAP
            if self.explainer and prediction == 1:
                try:
                    shap_values = self.explainer.shap_values(feature_df)
                    # For binary classification, shap_values[1] is the positive class (phishing)
                    # In some newer shap versions, it returns an array of shape (batch, features, classes)
                    if isinstance(shap_values, list):
                        pos_shap = shap_values[1][0]
                    else:
                        if len(shap_values.shape) == 3:
                            pos_shap = shap_values[0][:, 1]
                        else:
                            pos_shap = shap_values[0]
                        
                    # Map to feature names
                    feature_names = feature_df.columns.tolist()
                    contributions = list(zip(feature_names, pos_shap))
                    
                    # Sort by highest contribution to "Phishing"
                    contributions.sort(key=lambda x: x[1], reverse=True)
                    
                    # Take top 3 positive contributors
                    for feat, val in contributions[:3]:
                        if val > 0:
                            explanation.append({"feature": feat, "importance": float(val)})
                except Exception as e:
                    logger.error(f"SHAP explanation failed: {e}")
                    explanation_limitation = "Failed to generate feature attributions."
        
        return {
            'prediction': prediction,
            'confidence': confidence,
            'model_name': type(self.model).__name__,
            'is_calibrated': False,
            'model_explanation': explanation,
            'explanation_limitation': explanation_limitation
        }
