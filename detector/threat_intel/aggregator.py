import time
from typing import List, Dict, Any
from .providers.urlhaus import URLhausProvider
from .providers.google_safe_browsing import GoogleSafeBrowsingProvider

class ThreatIntelAggregator:
    def __init__(self, cache_ttl_seconds: int = 3600):
        # Initialize providers
        self.providers = [
            URLhausProvider(),
            GoogleSafeBrowsingProvider()
        ]
        self.cache = {}
        self.cache_ttl = cache_ttl_seconds

    def analyze(self, url: str) -> List[Dict[str, Any]]:
        """
        Query all active providers for the given URL.
        Uses an in-memory cache to respect rate limits and reduce latency.
        """
        current_time = time.time()
        
        # Check cache
        if url in self.cache:
            cached_data, timestamp = self.cache[url]
            if current_time - timestamp < self.cache_ttl:
                return cached_data

        results = []
        for provider in self.providers:
            result = provider.check_url(url)
            # Serialize enums for easy transport
            result['status'] = result['status'].value 
            results.append(result)
            
        # Update cache
        self.cache[url] = (results, current_time)
        return results
