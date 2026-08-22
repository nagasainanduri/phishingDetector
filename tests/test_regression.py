import sys
import os
import pickle
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scripts.feature_extraction import extract_features

def test_regression():
    try:
        with open('models/phishing_detector.pkl', 'rb') as f:
            model = pickle.load(f)
    except FileNotFoundError:
        print("Model file not found. Run training script first.")
        return

    # Deterministic test set
    test_data = [
        {"url": "https://experts.wahooas.org/js/js/4/", "expected": 1},
        {"url": "https://ibri.org/del/", "expected": 1},
        {"url": "https://google.com", "expected": 0},
        {"url": "http://example.com", "expected": 0}
    ]

    print("Running Baseline Regression Test")
    print("-" * 50)
    
    passed = 0
    failed = 0

    for item in test_data:
        url = item["url"]
        expected = item["expected"]
        
        features = extract_features(url)
        if not features:
            print(f"Failed to extract features for {url}")
            failed += 1
            continue
            
        feature_df = pd.DataFrame([features])
        # Add missing features if any
        for col in ['has_ip', 'https', 'num_dots', 'num_slashes', 'has_query', 'domain_length', 'tld_length', 'dns_record', 'has_at', 'has_dash', 'has_subdomain']:
            if col not in feature_df:
                feature_df[col] = 0
        
        # Ensure column order matches the model
        if hasattr(model, "feature_names_in_"):
            feature_df = feature_df[model.feature_names_in_]
                
        prediction = model.predict(feature_df)[0]
        result = "PASS" if prediction == expected else f"FAIL (Expected {expected}, got {prediction})"
        
        print(f"URL: {url} -> {result}")
        if prediction == expected:
            passed += 1
        else:
            failed += 1

    print("-" * 50)
    print(f"Total: {len(test_data)}, Passed: {passed}, Failed: {failed}")
    if failed > 0:
        sys.exit(1)

if __name__ == "__main__":
    test_regression()
