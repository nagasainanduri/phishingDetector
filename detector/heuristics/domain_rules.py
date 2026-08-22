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

def check_typosquatting(url, parsed):
    common_domains = ['paypal.com', 'apple.com', 'google.com', 'microsoft.com', 'amazon.com']
    netloc = parsed.netloc.lower()
    
    for d in common_domains:
        if d in netloc and netloc != d and not netloc.endswith("." + d):
            return RuleMatch(
                rule_id="DOM_004",
                severity=Severity.HIGH,
                description="Possible Typosquatting/Homoglyph Indicator.",
                evidence={"popular_domain_found": d, "actual_domain": netloc}
            )
    return None

TRUSTED_DOMAINS = {
    'github.com', 'google.com', 'microsoft.com', 'apple.com', 'amazon.com',
    'stackoverflow.com', 'linkedin.com', 'twitter.com', 'facebook.com', 'youtube.com'
}

def check_allowlist(url, parsed):
    if not parsed.hostname:
        return None
    hostname = parsed.hostname.lower()
    if any(hostname == domain or hostname.endswith('.' + domain) for domain in TRUSTED_DOMAINS):
        return RuleMatch(
            rule_id="DOM_005",
            severity=Severity.SAFE,
            description="Domain matches a highly trusted allowlist.",
            evidence={"hostname": hostname}
        )
    return None
