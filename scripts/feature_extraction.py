import pickle
import re
import socket
from urllib.parse import urlparse
import logging
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_cache():
    """Load feature cache from file."""
    cache_path = 'models/feature_cache.pkl'
    try:
        if os.path.exists(cache_path):
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        return {}
    except Exception as e:
        logger.error(f"Error loading cache: {type(e).__name__} - {str(e)}")
        return {}

def save_cache(cache):
    """Save feature cache to file."""
    cache_path = 'models/feature_cache.pkl'
    try:
        os.makedirs('model', exist_ok=True)
        with open(cache_path, 'wb') as f:
            pickle.dump(cache, f)
        logger.info(f"Cache saved to {cache_path}")
    except Exception as e:
        logger.error(f"Error saving cache: {type(e).__name__} - {str(e)}")

def extract_features(url, batch_cache=None):
    """Extract URL-based features from a URL, using batch_cache for in-memory updates."""
    if not url or not isinstance(url, str):
        return None

    # Check batch cache first, then global cache
    if batch_cache is not None and url in batch_cache:
        return batch_cache[url]

    cache = load_cache()
    if url in cache:
        return cache[url]

    try:
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        path = parsed.path
        query = parsed.query
        features = {}

        # Basic URL features
        features['url_length'] = len(url)
        features['has_ip'] = 1 if re.match(r'^(\d{1,3}\.){3}\d{1,3}$', domain) else 0
        features['https'] = 1 if parsed.scheme == 'https' else 0
        features['num_dots'] = domain.count('.')
        features['num_slashes'] = url.count('/')
        features['has_at'] = 1 if '@' in url else 0
        features['has_dash'] = 1 if '-' in domain else 0
        features['has_query'] = 1 if query else 0
        features['domain_length'] = len(domain)
        features['tld_length'] = len(domain.split('.')[-1]) if '.' in domain else 0
        features['has_subdomain'] = 1 if len(domain.split('.')) > 2 else 0

        # DNS record check
        try:
            socket.gethostbyname(domain)
            features['dns_record'] = 1
        except socket.gaierror:
            features['dns_record'] = 0

        # Update batch cache
        if batch_cache is not None:
            batch_cache[url] = features

        return features
    except Exception as e:
        logger.error(f"Error extracting features for {url}: {type(e).__name__} - {str(e)}")
        return None