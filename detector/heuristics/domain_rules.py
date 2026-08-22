import re
from .scoring import RuleMatch, Severity

def check_excessive_subdomains(url, parsed):
    if not parsed.hostname:
        return None
    parts = parsed.hostname.split('.')
    if len(parts) > 3 and not re.match(r'^(\d{1,3}\.){3}\d{1,3}$', parsed.hostname):
        return RuleMatch(
            rule_id="DOM_001",
            severity=Severity.MEDIUM,
            description="Excessive Subdomains. Cloud infrastructure (AWS, Azure) heavily utilize deeply nested subdomains natively. Proceed with caution.",
            evidence={"subdomain_count": len(parts) - 2}
        )
    return None

def check_punycode(url, parsed):
    if 'xn--' in parsed.netloc.lower():
        return RuleMatch(
            rule_id="DOM_002",
            severity=Severity.MEDIUM,
            description="Punycode domain detected (possible homograph attack).",
            evidence={"netloc": parsed.netloc}
        )
    return None

def check_url_shortener(url, parsed):
    shorteners = ['bit.ly', 'tinyurl.com', 't.co', 'goo.gl', 'ow.ly', 'is.gd', 'buff.ly']
    for s in shorteners:
        if s in parsed.netloc.lower():
            return RuleMatch(
                rule_id="DOM_003",
                severity=Severity.LOW,
                description="URL Shortener Service detected. Abused by phishers, but extensively used legitimately (e.g., Twitter, marketing).",
                evidence={"shortener": s}
            )
    return None

def check_brand_impersonation(url, parsed, brand_findings=None):
    if not brand_findings:
        return None
        
    observed = brand_findings["observed_domain"]
    expected = brand_findings["expected_domains"]
    
    # Check if observed domain is exactly one of the expected, or a subdomain of it
    if any(observed == d or observed.endswith('.' + d) for d in expected):
        # Legitimate domain!
        return None
        
    return RuleMatch(
        rule_id="DOM_004",
        severity=Severity.CRITICAL,
        description=f"Brand Impersonation! Highly similar to {brand_findings['possible_brand']} ({brand_findings['similarity_score']:.2f}).",
        evidence={
            "expected": expected,
            "observed": observed,
            "similarity": brand_findings['similarity_score'],
            "brand_evidence": brand_findings['evidence']
        }
    )

import os

# Load Tranco Top 100k once at module initialization
TRANCO_TOP_DOMAINS = set()
TRANCO_FILE = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'tranco_top_100k.txt')
if os.path.exists(TRANCO_FILE):
    with open(TRANCO_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            domain = line.strip().lower()
            if domain:
                TRANCO_TOP_DOMAINS.add(domain)

# Fallback top domains in case file is missing
TRUSTED_DOMAINS = {
    'github.com', 'google.com', 'microsoft.com', 'apple.com', 'amazon.com',
    'stackoverflow.com', 'linkedin.com', 'twitter.com', 'facebook.com', 'youtube.com'
}

def check_top_domain(url, parsed):
    if not parsed.hostname:
        return None
    hostname = parsed.hostname.lower()
    
    # Check exact match against Tranco set first (O(1))
    if hostname in TRANCO_TOP_DOMAINS:
        return RuleMatch(
            rule_id="DOM_005",
            severity=Severity.SAFE,
            description="Domain is ranked in the Global Top 100,000 (Tranco).",
            evidence={"hostname": hostname, "source": "Tranco"}
        )
        
    # Check fallback/subdomains against hardcoded trusted domains
    if any(hostname == domain or hostname.endswith('.' + domain) for domain in TRUSTED_DOMAINS):
        return RuleMatch(
            rule_id="DOM_005",
            severity=Severity.SAFE,
            description="Domain matches a highly trusted allowlist.",
            evidence={"hostname": hostname, "source": "Fallback Allowlist"}
        )
        
    # Check if a subdomain belongs to a Tranco top domain
    # Example: abc.github.io -> check github.io
    parts = hostname.split('.')
    if len(parts) > 2:
        # Check root domain (e.g., github.io)
        root_domain = f"{parts[-2]}.{parts[-1]}"
        if root_domain in TRANCO_TOP_DOMAINS:
            return RuleMatch(
                rule_id="DOM_005",
                severity=Severity.SAFE,
                description="Root domain is ranked in the Global Top 100,000 (Tranco).",
                evidence={"root_domain": root_domain, "source": "Tranco"}
            )
            
    return None
