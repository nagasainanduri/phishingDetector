from enum import Enum
from typing import Dict, Any

class ThreatStatus(Enum):
    MALICIOUS = "MALICIOUS"
    SAFE = "SAFE"
    UNKNOWN = "UNKNOWN"
    UNAVAILABLE = "UNAVAILABLE"

class ThreatIntelProvider:
    """
    Abstract base class for all threat intelligence providers.
    """
    def __init__(self, name: str, timeout: float = 2.0):
        self.name = name
        self.timeout = timeout
        
    def check_url(self, url: str) -> Dict[str, Any]:
        """
        Public interface that handles error boundary.
        Returns a dictionary containing the status and provider details.
        """
        if not self.is_configured():
            return {
                "provider": self.name,
                "status": ThreatStatus.UNAVAILABLE,
                "reason": "Not configured (Missing API key or disabled)"
            }
            
        try:
            status, metadata = self._query(url)
            return {
                "provider": self.name,
                "status": status,
                "metadata": metadata
            }
        except Exception as e:
            # Catch timeouts, connection errors, json decoding errors, etc.
            return {
                "provider": self.name,
                "status": ThreatStatus.UNAVAILABLE,
                "reason": f"Provider failed: {str(e)}"
            }

    def is_configured(self) -> bool:
        """
        Check if the provider has all required configuration (e.g. API keys).
        """
        return True

    def _query(self, url: str) -> tuple[ThreatStatus, Dict[str, Any]]:
        """
        Internal implementation for querying the external API.
        Must be implemented by subclasses.
        Returns (ThreatStatus, metadata_dict).
        """
        raise NotImplementedError("Providers must implement _query")
