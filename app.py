import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__)))  # Add project root to sys.path
from flask import Flask, render_template, request, jsonify
import pickle
import pandas as pd
from datetime import datetime
from scripts.feature_extraction import extract_features
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import logging
from cachetools import TTLCache
import threading
import warnings

# Suppress Flask-Limiter in-memory storage warning for dev
warnings.filterwarnings("ignore", category=UserWarning, module="flask_limiter")

app = Flask(__name__)

# Create logs and data directories
os.makedirs('logs', exist_ok=True)
os.makedirs('data', exist_ok=True)

# Rate limiting setup
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["100 per minute"]
)

# Cache for predictions (TTL: 1 hour)
cache = TTLCache(maxsize=1000, ttl=3600)
cache_lock = threading.Lock()

# Structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

MODEL_PATH = 'models/phishing_detector.pkl'
NEW_URLS_PATH = 'data/new_urls.csv'

try:
    with open(MODEL_PATH, 'rb') as file:
        model = pickle.load(file)
except FileNotFoundError:
    logger.error("Model file not found. Please run 'python scripts/train_model.py' first.")
    exit(1)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
@limiter.limit("10 per second")
def predict():
    global model
    data = request.form if request.form else request.json
    urls = data.get('urls') if isinstance(data.get('urls'), list) else [data.get('url')]
    if not urls:
        logger.warning("No URLs provided in request")
        return jsonify({'error': 'No URLs provided'}), 400

    results = []
    new_urls = []

    try:
        for url in urls:
            url = url.strip()
            # Check cache
            with cache_lock:
                if url in cache:
                    logger.info(f"Cache hit for URL: {url}")
                    results.append(cache[url])
                    continue

            # Predict
            try:
                features = extract_features(url)
                if features is None:
                    logger.error(f"Feature extraction failed for URL: {url}")
                    results.append({'url': url, 'error': 'Unable to verify this URL'})
                    continue
                feature_df = pd.DataFrame([features])
                prediction = model.predict(feature_df)[0]
                confidence = model.predict_proba(feature_df)[0][prediction]
                result = 'Phishing' if prediction == 1 else 'Legitimate'
                result_dict = {
                    'url': url,
                    'result': result,
                    'confidence': round(confidence * 100, 2)
                }
                results.append(result_dict)

                # Log to new_urls.csv
                new_urls.append({
                    'timestamp': datetime.now().isoformat(),
                    'url': url,
                    'result': result,
                    'confidence': confidence
                })

                # Cache result
                with cache_lock:
                    cache[url] = result_dict
            except Exception as e:
                logger.error(f"Error processing URL {url}: {str(e)}")
                results.append({'url': url, 'error': f'Unable to verify this URL: {str(e)}'})

        # Save to new_urls.csv
        if new_urls:
            new_urls_df = pd.DataFrame(new_urls)
            if os.path.exists(NEW_URLS_PATH):
                new_urls_df.to_csv(NEW_URLS_PATH, mode='a', header=False, index=False)
            else:
                new_urls_df.to_csv(NEW_URLS_PATH, mode='w', header=True, index=False)
            logger.info(f"Logged {len(new_urls)} URLs to {NEW_URLS_PATH}")

        return jsonify(results)

    except Exception as e:
        logger.error(f"Error analyzing URLs: {str(e)}")
        return jsonify({'error': f'Error analyzing URLs: {str(e)}'}), 500

@app.route('/api/predict', methods=['POST'])
@limiter.limit("10 per second")
def api_predict():
    return predict()

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)