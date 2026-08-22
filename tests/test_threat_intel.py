import unittest
from unittest.mock import patch
from detector.threat_intel.base import ThreatStatus
from detector.threat_intel.aggregator import ThreatIntelAggregator
from detector.heuristics.online_rules import check_threat_intel

class TestThreatIntel(unittest.TestCase):
    def setUp(self):
        self.aggregator = ThreatIntelAggregator(cache_ttl_seconds=3600)

    @patch('detector.threat_intel.providers.urlhaus.requests.post')
    def test_urlhaus_malicious(self, mock_post):
        # Mock URLhaus response
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {
            "query_status": "ok",
            "tags": ["phishing"],
            "threat": "malware_download"
        }
        
        # Test just the URLhaus provider
        provider = self.aggregator.providers[0] 
        res = provider.check_url("http://malicious.com")
        self.assertEqual(res["status"], ThreatStatus.MALICIOUS)
        self.assertIn("tags", res["metadata"])

    @patch('detector.threat_intel.providers.urlhaus.requests.post')
    def test_urlhaus_timeout_graceful_fail(self, mock_post):
        # Mock a timeout
        mock_post.side_effect = Exception("Connection Timeout")
        
        provider = self.aggregator.providers[0]
        res = provider.check_url("http://slow-site.com")
        
        # Should gracefully return UNAVAILABLE, not crash
        self.assertEqual(res["status"], ThreatStatus.UNAVAILABLE)
        self.assertIn("Connection Timeout", res["reason"])

    def test_google_safe_browsing_no_key(self):
        # Ensure it gracefully disables itself if no API key is present
        with patch.dict('os.environ', {}, clear=True):
            # re-initialize to pick up empty env
            from detector.threat_intel.providers.google_safe_browsing import GoogleSafeBrowsingProvider
            gsb = GoogleSafeBrowsingProvider()
            res = gsb.check_url("http://test.com")
            
            self.assertEqual(res["status"], ThreatStatus.UNAVAILABLE)
            self.assertEqual(res["reason"], "Not configured (Missing API key or disabled)")

    def test_heuristic_rule_triggers_on_malicious(self):
        # Create a fake findings list
        findings = [
            {"provider": "URLhaus", "status": "MALICIOUS", "metadata": {}},
            {"provider": "Google Safe Browsing", "status": "UNAVAILABLE", "reason": "No Key"}
        ]
        
        match = check_threat_intel("http://bad.com", None, findings)
        self.assertIsNotNone(match)
        self.assertEqual(match.rule_id, "ONL_002")
        self.assertIn("URLhaus", match.description)

    def test_heuristic_rule_ignores_safe(self):
        # Create a fake findings list
        findings = [
            {"provider": "URLhaus", "status": "UNKNOWN", "metadata": {}},
            {"provider": "Google Safe Browsing", "status": "SAFE", "reason": "Not flagged"}
        ]
        
        match = check_threat_intel("http://good.com", None, findings)
        self.assertIsNone(match)

if __name__ == '__main__':
    unittest.main()
