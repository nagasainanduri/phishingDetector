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
                result_dict = detector.analyze(url)
                results.append(result_dict)
                
                # If there's no error, we log it
                if 'error' not in result_dict and result_dict.get('result') != 'Error':
                    # Log to new_urls.csv
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

@app.route('/api/predict', methods=['POST'])
@limiter.limit("10 per second")
def api_predict():
    return predict()

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)