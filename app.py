import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__)))  # Add project root to sys.path
from flask import Flask, render_template, request, jsonify
import pandas as pd
from datetime import datetime
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import logging
from cachetools import TTLCache
import threading
import warnings
from detector import PhishingDetector
from flask_cors import CORS

# Suppress Flask-Limiter in-memory storage warning for dev
warnings.filterwarnings("ignore", category=UserWarning, module="flask_limiter")

app = Flask(__name__)
CORS(app)

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

NEW_URLS_PATH = 'data/new_urls.csv'

# Initialize detector
detector = PhishingDetector()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
@limiter.limit("10 per second")
def predict():
    data = request.form if request.form else request.json
    urls = data.get('urls') if isinstance(data.get('urls'), list) else [data.get('url')]
    page_signals = data.get('page_signals')
    
    # Extract privacy configurations
    privacy_mode = data.get('privacy_mode', 'local_only')
    telemetry = data.get('telemetry', False)
    
    if not urls or not urls[0]:
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
                result_dict = detector.analyze(url, page_signals=page_signals, privacy_mode=privacy_mode)
                results.append(result_dict)
                
                # Only log to CSV if telemetry is explicitly enabled (Data Minimization)
                if telemetry and 'error' not in result_dict and result_dict.get('result') != 'Error':
                    new_urls.append({
                        'timestamp': datetime.now().isoformat(),
                        'url': url,
                        'result': result_dict['result'],
                        'confidence': result_dict['confidence'] / 100.0  # normalize back for CSV
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

import hashlib

@app.route('/api/feedback', methods=['POST'])
@limiter.limit("20 per minute")
def submit_feedback():
    data = request.json
    if not data or not data.get('url'):
        return jsonify({'error': 'Missing URL'}), 400
        
    url = data.get('url')
    feedback_type = data.get('feedback_type')
    share_raw_url = data.get('share_raw_url', False)
    risk_score = data.get('risk_score', 0)
    prediction = data.get('prediction', 'UNKNOWN')
    
    # Privacy mechanism: Hash the URL if the user declined sharing raw URL or if it's just 'correct'
    if not share_raw_url or feedback_type == 'correct':
        url_identifier = "HASHED:" + hashlib.sha256(url.encode()).hexdigest()
    else:
        url_identifier = url
        
    feedback_entry = {
        'timestamp': datetime.now().isoformat(),
        'url_identifier': url_identifier,
        'feedback_type': feedback_type,
        'risk_score': risk_score,
        'prediction': prediction
    }
    
    feedback_path = 'data/feedback.csv'
    df = pd.DataFrame([feedback_entry])
    if os.path.exists(feedback_path):
        df.to_csv(feedback_path, mode='a', header=False, index=False)
    else:
        df.to_csv(feedback_path, mode='w', header=True, index=False)
        
    logger.info(f"Feedback logged: {feedback_type} for {url_identifier}")
    return jsonify({'status': 'success'})

@app.route('/api/predict', methods=['POST'])
@limiter.limit("10 per second")
def api_predict():
    return predict()

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)