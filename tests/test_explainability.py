import unittest
from unittest.mock import patch, MagicMock
from detector.models.predictor import PhishingPredictor

class TestExplainability(unittest.TestCase):
    def test_random_forest_explainability(self):
        # The default model is RandomForest. Let's see if SHAP runs successfully
        predictor = PhishingPredictor()
        
        # Mock features that would strongly suggest phishing (high url length, etc)
        features = {
            'url_length': 150,
            'has_ip': 1,
            'https': 0,
            'num_dots': 5,
            'num_slashes': 5,
            'has_at': 1,
            'has_dash': 1,
            'has_query': 1,
            'domain_length': 50,
            'tld_length': 3,
            'has_subdomain': 1,
            'dns_record': 0
        }
        
        # Test it - ensure it doesn't crash and returns the correct struct
        result = predictor.predict("http://bad.com", features)
        
        self.assertIn('model_explanation', result)
        self.assertIn('explanation_limitation', result)
        
        if result['prediction'] == 1 and predictor.explainer is not None:
            # SHAP should have returned something
            self.assertTrue(len(result['model_explanation']) > 0)
            self.assertIsNone(result['explanation_limitation'])
            
            # Check structure
            top_feat = result['model_explanation'][0]
            self.assertIn('feature', top_feat)
            self.assertIn('importance', top_feat)

    @patch('detector.models.predictor.type')
    def test_cnn_limitation(self, mock_type):
        # Force it to think the model is CharCNN
        predictor = PhishingPredictor()
        predictor.model = MagicMock()
        mock_type.return_value.__name__ = 'CharCNNClassifier'
        
        result = predictor.predict("http://bad.com")
        
        self.assertEqual(result['model_explanation'], [])
        self.assertIsNotNone(result['explanation_limitation'])
        self.assertIn("not supported for sequence-based", result['explanation_limitation'])

if __name__ == '__main__':
    unittest.main()
