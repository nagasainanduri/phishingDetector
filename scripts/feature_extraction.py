import requests
import whois
import re
import os
from urllib.parse import urlparse
import logging
import pickle

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Cache file for storing extracted features
# This file will be created in the 'model' directory
CACHE_FILE = 'model/feature_cache.pkl'
cache={}

def load_cache():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, 'rb') as f:
            return pickle.load(f)
    return {}

def save_cache():
    if os.path.exists(CACHE_FILE):
        os.remove(CACHE_FILE)
    with open(CACHE_FILE, 'wb') as f:
        pickle.dump(cache, f)

def extract_features(url):
    """Extract features from a URL."""
    global cache
    if not cache:
        cache = load_cache()
        
    if url in cache:
        logger.info(f"Using cached features for {url}")
        return cache[url]
    
    try:
        parsed_url = urlparse(url)
        features = {
            'has_ip': 1 if re.match(r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$', parsed_url.netloc) else 0,
            'url_length': len(url),
            'https': 1 if parsed_url.scheme == 'https' else 0,
            'num_dots': url.count('.'),
            'num_slashes': url.count('/'),
            'has_at': 1 if '@' in url else 0,
            'has_dash': 1 if '-' in url else 0,
            'has_query': 1 if parsed_url.query else 0,
            'domain_length': len(parsed_url.netloc),
            'tld_length': len(parsed_url.netloc.split('.')[-1]) if parsed_url.netloc else 0,
            'dns_record': 0,
            'num_anchors': 0,
            'external_anchors': 0,
            'num_forms': 0,
            'has_popup': 0,
            'meta_refresh': 0,
            'domain_age': -1
        }

        # DNS and WHOIS lookup
        try:
            w = whois.whois(parsed_url.netloc, timeout=5)
            features['dns_record'] = 1
            if w.creation_date:
                creation_date = w.creation_date[0] if isinstance(w.creation_date, list) else w.creation_date
                features['domain_age'] = (datetime.now() - creation_date).days
        except Exception:
            features['dns_record'] = 0
            features['domain_age'] = -1

        # Webpage content analysis
        try:
            response = requests.get(url, timeout=5, headers={'User-Agent': 'Mozilla/5.0'})
            content = response.text.lower()
            features['num_anchors'] = content.count('<a ')
            features['external_anchors'] = content.count('href="http') - content.count(f'href="http://{parsed_url.netloc}') - content.count(f'href="https://{parsed_url.netloc}')
            features['num_forms'] = content.count('<form')
            features['has_popup'] = 1 if 'window.open' in content else 0
            features['meta_refresh'] = 1 if '<meta http-equiv="refresh"' in content else 0
        except Exception:
            pass
        
        cache[url] = features
        save_cache(cache)
        return features
    except Exception as e:
        logger.warning(f"Feature extraction failed for {url}: {str(e)}")
        return None