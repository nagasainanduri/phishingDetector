import os
import requests
from ..base import ThreatIntelProvider, ThreatStatus

class GoogleSafeBrowsingProvider(ThreatIntelProvider):
    def __init__(self):
        super().__init__(name="Google Safe Browsing")
        self.api_key = os.environ.get("GSB_API_KEY")
        self.api_url = "https://safebrowsing.googleapis.com/v4/threatMatches:find"

    def is_configured(self) -> bool:
        return bool(self.api_key)

    def _query(self, url: str):
        payload = {
            "client": {
                "clientId": "phishguard",
                "clientVersion": "1.0.0"
            },
            "threatInfo": {
                "threatTypes": ["MALWARE", "SOCIAL_ENGINEERING", "UNWANTED_SOFTWARE"],
                "platformTypes": ["ANY_PLATFORM"],
                "threatEntryTypes": ["URL"],
                "threatEntries": [
                    {"url": url}
                ]
            }
        }
        
        params = {'key': self.api_key}
        response = requests.post(self.api_url, params=params, json=payload, timeout=self.timeout)
        response.raise_for_status()
        
        result = response.json()
        matches = result.get('matches')
        
        if matches:
            threat_types = [m.get("threatType") for m in matches]
            return ThreatStatus.MALICIOUS, {"threat_types": threat_types}
            
        return ThreatStatus.SAFE, {"reason": "Not flagged by Google Safe Browsing"}
