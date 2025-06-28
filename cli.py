import argparse
import pandas as pd
import pickle
from scripts.feature_extraction import extract_features

def load_model():
    try:
        with open('models/phishing_detector.pkl', 'rb') as file:
            return pickle.load(file)
    except FileNotFoundError:
        print("Error: Model file not found. Run 'python scripts/train_model.py' first.")
        exit(1)

def predict_url(url, model):
    try:
        features = extract_features(url)
        feature_df = pd.DataFrame([features])
        prediction = model.predict(feature_df)[0]
        confidence = model.predict_proba(feature_df)[0][prediction]
        result = 'Phishing' if prediction == 1 else 'Legitimate'
        return {'url': url, 'result': result, 'confidence': round(confidence * 100, 2)}
    except Exception as e:
        return {'url': url, 'result': 'Error', 'confidence': 0, 'error': str(e)}

def main():
    parser = argparse.ArgumentParser(description='Phishing Web Detector CLI')
    parser.add_argument('--url', type=str, help='Single URL to analyze')
    parser.add_argument('--file', type=str, help='Path to a file with URLs (one per line)')
    args = parser.parse_args()

    model = load_model()

    if args.url:
        result = predict_url(args.url, model)
        print(f"URL: {result['url']}")
        print(f"Result: {result['result']}")
        print(f"Confidence: {result['confidence']}%")
        if 'error' in result:
            print(f"Error: {result['error']}")
    elif args.file:
        try:
            with open(args.file, 'r') as f:
                urls = [line.strip() for line in f if line.strip()]
            for url in urls:
                result = predict_url(url, model)
                print(f"URL: {result['url']}")
                print(f"Result: {result['result']}")
                print(f"Confidence: {result['confidence']}%")
                if 'error' in result:
                    print(f"Error: {result['error']}")
                print("-" * 50)
        except FileNotFoundError:
            print(f"Error: File '{args.file}' not found.")
    else:
        print("Error: Please provide a --url or --file argument.")
        parser.print_help()

if __name__ == '__main__':
    main()