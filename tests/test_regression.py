import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from detector import PhishingDetector

def test_regression():
    try:
        detector = PhishingDetector()
    except Exception as e:
        print(f"Model file not found or failed to load: {e}")
        return

    # Deterministic test set
    test_data = [
        {"url": "https://experts.wahooas.org/js/js/4/", "expected": "WARN"}, # High ML, low heuristics may map to WARN or BLOCK depending on actual config
        {"url": "https://ibri.org/del/", "expected": "ALLOW"},
        {"url": "https://google.com", "expected": "ALLOW"},
        {"url": "http://example.com", "expected": "ALLOW"}
    ]

    print("Running Baseline Regression Test")
    print("-" * 50)
    
    passed = 0
    failed = 0

    for item in test_data:
        url = item["url"]
        expected = item["expected"]
        
        result_dict = detector.analyze(url)
        if result_dict.get('result') == 'Error':
            print(f"Failed to analyze URL {url}: {result_dict.get('error')}")
            failed += 1
            continue
            
        prediction = result_dict['result']
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
