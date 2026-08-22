import json
import os
from urllib.parse import urlparse
from .similarity import decode_punycode, normalize_homoglyphs, compute_similarity

class BrandDetector:
    def __init__(self, config_path=None):
        if config_path is None:
            config_path = os.path.join(os.path.dirname(__file__), 'brands.json')
            
        with open(config_path, 'r', encoding='utf-8') as f:
            self.brands = json.load(f)

    def analyze(self, url: str) -> dict:
        """
        Analyzes a URL for brand impersonation.
        Returns a structured finding dict if a brand is detected (similarity >= 0.8),
        otherwise returns None.
        """
        try:
            if not url.startswith(('http://', 'https://')):
                url = 'http://' + url
            parsed = urlparse(url)
            netloc = parsed.netloc.lower()
        except Exception:
            return None

        if not netloc:
            return None

        # 1. Decode punycode
        decoded_netloc = decode_punycode(netloc)
        
        # 2. Normalize homoglyphs
        normalized_netloc = normalize_homoglyphs(decoded_netloc)
        
        # 3. Split domain into parts to find the base word
        # (e.g., login.paypa1.com -> ['login', 'paypa1', 'com'])
        parts = normalized_netloc.split('.')
        
        best_match = None
        highest_score = 0.0
        matched_brand = None
        
        # Compare each part of the domain against known brands
        for part in parts:
            if len(part) < 3:
                continue # Skip very short parts like 'co', 'uk', 'com'
                
            for brand_name, data in self.brands.items():
                score = compute_similarity(part, brand_name)
                if score > highest_score:
                    highest_score = score
                    best_match = part
                    matched_brand = brand_name
        
        if highest_score >= 0.8 and matched_brand:
            evidence = []
            if 'xn--' in netloc:
                evidence.append("Punycode decoded")
            if decoded_netloc != normalized_netloc:
                evidence.append("Homoglyphs normalized")
                
            return {
                "possible_brand": matched_brand,
                "observed_domain": netloc,
                "expected_domains": self.brands[matched_brand]["known_domains"],
                "similarity_score": highest_score,
                "matched_part": best_match,
                "evidence": evidence
            }
            
        return None
