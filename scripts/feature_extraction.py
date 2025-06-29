import requests
import whois
import re
import os
from urllib.parse import urlparse
import logging
import pickle
from datetime import datetime
from tenacity import retry, stop_after_attempt, wait_fixed

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Cache file for storing extracted features
CACHE_FILE = 'models/feature_cache.pkl'
cache = {}

def load_cache():
    """Load the cache dictionary from a file."""
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            logger.warning(f"Failed to load cache from {CACHE_FILE}: {type(e).__name__} - {str(e)}")
            return {}
    return {}

def save_cache(cache):
    """Save the cache dictionary to a file."""
    try:
        os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
        if os.path.exists(CACHE_FILE):
            os.remove(CACHE_FILE)
        with open(CACHE_FILE, 'wb') as f:
            pickle.dump(cache, f)
        logger.info(f"Cache saved to {CACHE_FILE}")
    except Exception as e:
        logger.error(f"Failed to save cache to {CACHE_FILE}: {type(e).__name__} - {str(e)}")

@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def get_webpage(url):
    """Fetch webpage content with retries."""
    return requests.get(url, timeout=10, headers={'User-Agent': 'Mozilla/5.0'})

@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def get_whois(netloc):
    """Perform WHOIS lookup with retries."""
    return whois.whois(netloc, timeout=10)

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
            w = get_whois(parsed_url.netloc)
            features['dns_record'] = 1
            if w.creation_date:
                creation_date = w.creation_date[0] if isinstance(w.creation_date, list) else w.creation_date
                features['domain_age'] = (datetime.now() - creation_date).days
        except Exception as e:
            logger.debug(f"WHOIS lookup failed for {url}: {type(e).__name__} - {str(e)}")
            features['dns_record'] = 0
            features['domain_age'] = -1

        # Webpage content analysis
        try:
            response = get_webpage(url)
            content = response.text.lower()
            features['num_anchors'] = content.count('<a ')
            features['external_anchors'] = content.count('href="http') - content.count(f'href="http://{parsed_url.netloc}') - content.count(f'href="https://{parsed_url.netloc}')
            features['num_forms'] = content.count('<form')
            features['has_popup'] = 1 if 'window.open' in content else 0
            features['meta_refresh'] = 1 if '<meta http-equiv="refresh"' in content else 0
        except Exception as e:
            logger.debug(f"Webpage content analysis failed for {url}: {type(e).__name__} - {str(e)}")
        
        cache[url] = features
        save_cache(cache)
        return features
    except Exception as e:
        logger.warning(f"Feature extraction failed for {url}: {type(e).__name__} - {str(e)}")
        return None