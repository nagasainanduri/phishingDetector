import pickle
import re
import socket
from urllib.parse import urlparse
import logging
import os

logger = logging.getLogger(__name__)

def validate_url(url, kaggele_mode=False):
    """Validate if a string is a proper URL with a domain or IP."""
    if not url or not isinstance(url, str):
        return False
    try:
        #for kaggele dataset => this dataset does not have http:// or https:// in the URL
        if kaggele_mode and not url.startswith(('http://', 'https://')):
            url = f"https://{url}"
        parsed = urlparse(url)
        
        if not kaggele_mode and not parsed.scheme in ['http', 'https']:
            return False
        if not parsed.netloc:
            return False

        if re.match(r'^\d+$', parsed.netloc):
            return False  # Just a number, not a valid domain or IP
        
        # Accept domains or IPs
        if re.match(r'^([a-zA-Z0-9-]+\.)*[a-zA-Z0-9-]+\.[a-zA-Z]{2,}$|^(\d{1,3}\.){3}\d{1,3}$', parsed.netloc):
            return True
        return False
    except Exception:
        return False

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
        os.makedirs('models', exist_ok=True)
        with open(cache_path, 'wb') as f:
            pickle.dump(cache, f)
        logger.debug(f"Cache saved to {cache_path}")
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

        # Basic URL features (Existing)
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

        # New URL Features
        features['path_length'] = len(path)
        features['query_length'] = len(query)
        features['fragment_length'] = len(parsed.fragment)
        features['num_parameters'] = query.count('&') + 1 if query else 0
        
        # Ratio of non-alphanumeric chars
        alnum_count = sum(c.isalnum() for c in url)
        features['special_char_ratio'] = (len(url) - alnum_count) / len(url) if len(url) > 0 else 0
        features['has_encoding'] = 1 if re.search(r'%[0-9a-fA-F]{2}', url) else 0

        # New Domain Features
        domain_parts = domain.split('.')
        features['subdomain_depth'] = max(0, len(domain_parts) - 2)
        digit_count = sum(c.isdigit() for c in domain)
        features['domain_digit_ratio'] = digit_count / len(domain) if len(domain) > 0 else 0
        features['num_hyphens_domain'] = domain.count('-')
        features['has_punycode'] = 1 if 'xn--' in domain else 0
        
        # Extract TLD
        features['tld'] = domain_parts[-1] if len(domain_parts) > 1 and not features['has_ip'] else 'none'
        
        # Shannon Entropy of domain
        import math
        from collections import Counter
        p, lns = Counter(domain), float(len(domain))
        features['domain_entropy'] = -sum(count/lns * math.log(count/lns, 2) for count in p.values()) if lns > 0 else 0

        # New Structural Features
        features['has_explicit_port'] = 1 if ':' in parsed.netloc else 0
        features['url_depth'] = path.strip('/').count('/') + 1 if path.strip('/') else 0

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
