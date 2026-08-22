import requests
from ..base import ThreatIntelProvider, ThreatStatus

class URLhausProvider(ThreatIntelProvider):
    def __init__(self):
        super().__init__(name="URLhaus")
        self.api_url = "https://urlhaus-api.abuse.ch/v1/url/"

    def _query(self, url: str):
        data = {'url': url}
        
        # Free API, no auth required
        response = requests.post(self.api_url, data=data, timeout=self.timeout)
        response.raise_for_status()
        
        result = response.json()
        query_status = result.get('query_status')
        
        if query_status == 'ok':
            # It's in the database
            tags = result.get('tags', [])
            return ThreatStatus.MALICIOUS, {"tags": tags, "threat": result.get('threat')}
            
        elif query_status == 'no_results':
            return ThreatStatus.UNKNOWN, {"reason": "Not found in URLhaus database"}
            
        return ThreatStatus.UNAVAILABLE, {"reason": f"Unexpected status: {query_status}"}
