import unittest
from detector.brand.similarity import (
    decode_punycode,
    normalize_homoglyphs,
    levenshtein_distance,
    compute_similarity
)
from detector.brand.detector import BrandDetector
from detector.heuristics.domain_rules import check_brand_impersonation
from urllib.parse import urlparse
from detector.heuristics.scoring import Severity

class TestBrandDetector(unittest.TestCase):
    def setUp(self):
        self.detector = BrandDetector()

    def test_similarity_functions(self):
        self.assertEqual(levenshtein_distance("paypal", "paypal"), 0)
        self.assertEqual(levenshtein_distance("paypal", "paypa1"), 1)
        
        sim = compute_similarity("paypal", "paypa1")
        self.assertAlmostEqual(sim, 0.8333, places=4)
        
    def test_homoglyphs_and_punycode(self):
        # 'а' is cyrillic
        cyrillic_paypal = "pаypal"
        normalized = normalize_homoglyphs(cyrillic_paypal)
        self.assertEqual(normalized, "paypal")
        
        # punycode for cyrillic 'а' in paypal
        punycode_domain = "xn--pypal-4ve.com"
        decoded = decode_punycode(punycode_domain)
        self.assertTrue('а' in decoded)

    def test_detector_exact_match(self):
        res = self.detector.analyze("https://paypal.com")
        self.assertIsNotNone(res)
        self.assertEqual(res['possible_brand'], "paypal")
        self.assertEqual(res['similarity_score'], 1.0)
        
    def test_detector_typosquatting(self):
        res = self.detector.analyze("http://paypa1.com")
        self.assertIsNotNone(res)
        self.assertEqual(res['possible_brand'], "paypal")
        self.assertGreaterEqual(res['similarity_score'], 0.8)
        
    def test_detector_unrelated(self):
        res = self.detector.analyze("https://example.com")
        self.assertIsNone(res)
        
    def test_heuristic_rule(self):
        # 1. Exact match (Legitimate)
        exact_res = self.detector.analyze("https://paypal.com")
        parsed1 = urlparse("https://paypal.com")
        match1 = check_brand_impersonation("https://paypal.com", parsed1, brand_findings=exact_res)
        self.assertIsNone(match1) # Should not flag as impersonation!
        
        # 2. Typosquatting (Impersonation)
        typo_res = self.detector.analyze("https://paypa1.com")
        parsed2 = urlparse("https://paypa1.com")
        match2 = check_brand_impersonation("https://paypa1.com", parsed2, brand_findings=typo_res)
        self.assertIsNotNone(match2)
        self.assertEqual(match2.rule_id, "DOM_004")
        self.assertEqual(match2.severity, Severity.CRITICAL)

if __name__ == '__main__':
    unittest.main()
