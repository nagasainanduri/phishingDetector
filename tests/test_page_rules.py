import unittest
from urllib.parse import urlparse
from detector.heuristics.scoring import Severity
from detector.heuristics.page_rules import (check_credential_collection, 
                                            check_cross_origin_action, 
                                            check_hidden_iframes)

class TestPageRules(unittest.TestCase):
    def test_check_credential_collection(self):
        # Should flag HTTP with password field
        url = "http://example.com/login"
        parsed = urlparse(url)
        signals = {"has_password_field": True}
        match = check_credential_collection(url, parsed, signals)
        self.assertIsNotNone(match)
        self.assertEqual(match.rule_id, "PAGE_001")
        self.assertEqual(match.severity, Severity.HIGH)

        # Should NOT flag HTTPS with password field
        url_https = "https://example.com/login"
        parsed_https = urlparse(url_https)
        match2 = check_credential_collection(url_https, parsed_https, signals)
        self.assertIsNone(match2)

    def test_check_cross_origin_action(self):
        url = "https://example.com/login"
        parsed = urlparse(url)
        
        # High/Critical risk if cross-origin action AND has login/password
        signals_critical = {
            "cross_origin_form_action": True,
            "has_login_form": True
        }
        match1 = check_cross_origin_action(url, parsed, signals_critical)
        self.assertIsNotNone(match1)
        self.assertEqual(match1.severity, Severity.CRITICAL)
        
        # Medium risk if just cross-origin form (could be a legit search form etc)
        signals_medium = {
            "cross_origin_form_action": True,
            "has_login_form": False
        }
        match2 = check_cross_origin_action(url, parsed, signals_medium)
        self.assertIsNotNone(match2)
        self.assertEqual(match2.severity, Severity.MEDIUM)

    def test_check_hidden_iframes(self):
        url = "https://example.com"
        parsed = urlparse(url)
        
        signals_bad = {"hidden_iframes": True}
        match = check_hidden_iframes(url, parsed, signals_bad)
        self.assertIsNotNone(match)
        self.assertEqual(match.severity, Severity.MEDIUM)
        
        signals_good = {"hidden_iframes": False}
        match2 = check_hidden_iframes(url, parsed, signals_good)
        self.assertIsNone(match2)

if __name__ == '__main__':
    unittest.main()
