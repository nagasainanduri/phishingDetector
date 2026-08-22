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
import hashlib

# Suppress Flask-Limiter in-memory storage warning for dev
warnings.filterwarnings("ignore", category=UserWarning, module="flask_limiter")

app = Flask(__name__)
CORS(app)

# 1. Request Size Limits (1 MB)
app.config['MAX_CONTENT_LENGTH'] = 1 * 1024 * 1024

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

# ==========================================
# 2. Structured Error Handlers
# ==========================================
def make_error_response(code, message):
    return jsonify({"error": {"code": code, "message": message}}), code

@app.errorhandler(400)
def bad_request(e):
    return make_error_response(400, "Bad Request")

@app.errorhandler(404)
def not_found(e):
    return make_error_response(404, "Endpoint not found")

@app.errorhandler(413)
def request_entity_too_large(e):
    return make_error_response(413, "Payload too large. Maximum size is 1MB.")

@app.errorhandler(429)
def ratelimit_handler(e):
    return make_error_response(429, "Rate limit exceeded. Please slow down.")

@app.errorhandler(500)
def internal_server_error(e):
    return make_error_response(500, "Internal server error")

# ==========================================
# Legacy Route
# ==========================================
@app.route('/')
def index():
    return render_template('index.html')

# ==========================================
# 4. Versioned API Endpoints
# ==========================================
@app.route('/api/v1/health', methods=['GET'])
def health_check():
    return jsonify({"status": "ok", "version": "1.0.0"})

@app.route('/api/v1/analyze', methods=['POST'])
@app.route('/api/v1/reputation', methods=['POST'])  # Alias for now
@limiter.limit("10 per second")
def analyze_url():
    data = request.json
    if not data:
        return make_error_response(400, "Invalid JSON payload")
        
    urls = data.get('urls') if isinstance(data.get('urls'), list) else [data.get('url')]
    page_signals = data.get('page_signals')
    
    privacy_mode = data.get('privacy_mode', 'local_only')
    telemetry = data.get('telemetry', False)
    
    if not urls or not urls[0]:
        logger.warning("No URLs provided in request")
        return make_error_response(400, "No URLs provided")
        
    # Input Validation
    for url in urls:
        if not isinstance(url, str):
            return make_error_response(400, "URLs must be strings")
        if len(url) > 2048:
            return make_error_response(400, "URL exceeds maximum length of 2048 characters")

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
                
                # Only log to CSV if telemetry is explicitly enabled
                if telemetry and 'error' not in result_dict and result_dict.get('result') != 'Error':
                    new_urls.append({
                        'timestamp': datetime.now().isoformat(),
                        'url': url,
                        'result': result_dict['result'],
                        'confidence': result_dict['confidence'] / 100.0
                    })

                # Cache result
                with cache_lock:
                    cache[url] = result_dict
            except Exception as e:
                logger.error(f"Error processing URL {url}: {str(e)}")
                # Do not crash the API, return a structured error for this item
                results.append({'url': url, 'error': f'Unable to verify this URL: {str(e)}'})

        # Save to new_urls.csv
        if new_urls:
            new_urls_df = pd.DataFrame(new_urls)
            if os.path.exists(NEW_URLS_PATH):
                new_urls_df.to_csv(NEW_URLS_PATH, mode='a', header=False, index=False)
            else:
                new_urls_df.to_csv(NEW_URLS_PATH, mode='w', header=True, index=False)

        return jsonify(results)

    except Exception as e:
        logger.error(f"Error analyzing URLs: {str(e)}")
        return make_error_response(500, "Internal server error during analysis")

@app.route('/api/v1/feedback', methods=['POST'])
@limiter.limit("20 per minute")
def submit_feedback():
    data = request.json
    if not data or not data.get('url'):
        return make_error_response(400, "Missing URL")
        
    url = data.get('url')
    if not isinstance(url, str):
        return make_error_response(400, "URL must be a string")
        
    feedback_type = data.get('feedback_type')
    share_raw_url = data.get('share_raw_url', False)
    risk_score = data.get('risk_score', 0)
    prediction = data.get('prediction', 'UNKNOWN')
    
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
    try:
        df = pd.DataFrame([feedback_entry])
        if os.path.exists(feedback_path):
            df.to_csv(feedback_path, mode='a', header=False, index=False)
        else:
            df.to_csv(feedback_path, mode='w', header=True, index=False)
    except Exception as e:
        logger.error(f"Failed to write feedback: {str(e)}")
        return make_error_response(500, "Failed to store feedback")
        
    logger.info(f"Feedback logged: {feedback_type} for {url_identifier}")
    return jsonify({'status': 'success'})

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)