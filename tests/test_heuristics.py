import unittest
import sys
import os

# Ensure the root directory is in the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from detector.heuristics import create_engine, Severity

class TestHeuristics(unittest.TestCase):
    def setUp(self):
        self.engine = create_engine()

    def test_ip_address_rule(self):
        matches = self.engine.evaluate('http://192.168.1.1/login')
        self.assertTrue(any(m.rule_id == 'URL_001' and m.severity == Severity.HIGH for m in matches))
        
        matches_clean = self.engine.evaluate('https://google.com')
        self.assertFalse(any(m.rule_id == 'URL_001' for m in matches_clean))

    def test_at_symbol_rule(self):
        matches = self.engine.evaluate('http://paypal.com@attacker.com/login')
        self.assertTrue(any(m.rule_id == 'URL_003' and m.severity == Severity.HIGH for m in matches))

    def test_excessive_subdomains(self):
        matches = self.engine.evaluate('http://a.b.c.d.example.com')
        self.assertTrue(any(m.rule_id == 'DOM_001' and m.severity == Severity.MEDIUM for m in matches))

    def test_punycode_rule(self):
        matches = self.engine.evaluate('http://xn--c1yn36f.com')
        self.assertTrue(any(m.rule_id == 'DOM_002' and m.severity == Severity.MEDIUM for m in matches))

    def test_url_shortener(self):
        matches = self.engine.evaluate('https://bit.ly/12345')
        self.assertTrue(any(m.rule_id == 'DOM_003' and m.severity == Severity.LOW for m in matches))

    def test_typosquatting(self):
        matches = self.engine.evaluate('http://paypal.com.login-update.xyz')
        self.assertTrue(any(m.rule_id == 'DOM_004' and m.severity == Severity.HIGH for m in matches))
        
        # False positive test: Should not trigger on actual paypal.com
        matches_clean = self.engine.evaluate('https://www.paypal.com')
        self.assertFalse(any(m.rule_id == 'DOM_004' for m in matches_clean))

    def test_suspicious_patterns(self):
        matches = self.engine.evaluate('http://example.com/login.php?cmd=execute')
        self.assertTrue(any(m.rule_id == 'URL_006' and m.severity == Severity.MEDIUM for m in matches))

    def test_clean_url(self):
        matches = self.engine.evaluate('https://github.com/microsoft/vscode')
        self.assertEqual(len(matches), 0)

    def test_edge_cases(self):
        # IP with explicit port
        matches = self.engine.evaluate('http://192.168.1.1:8080/admin')
        rule_ids = [m.rule_id for m in matches]
        self.assertIn('URL_001', rule_ids) # IP address
        self.assertIn('URL_004', rule_ids) # Explicit port
        self.assertIn('URL_006', rule_ids) # 'admin' in path
        
        # Heavy URL Encoding
        matches_enc = self.engine.evaluate('http://example.com/redirect?q=%20%20%20%20%20%20%20')
        self.assertTrue(any(m.rule_id == 'URL_005' and m.severity == Severity.LOW for m in matches_enc))
        
        # Invalid / Empty URL shouldn't crash
        matches_inv = self.engine.evaluate('not_a_url')
        self.assertEqual(len(matches_inv), 0)

if __name__ == '__main__':
    unittest.main()
