import re
from .scoring import RuleMatch, Severity

def check_ip_address(url, parsed):
    if parsed.hostname and re.match(r'^(\d{1,3}\.){3}\d{1,3}$', parsed.hostname):
        return RuleMatch(
            rule_id="URL_001",
            severity=Severity.HIGH,
            description="IP Address used instead of Domain Name.",
            evidence={"hostname": parsed.hostname}
        )
    return None

def check_excessive_length(url, parsed):
    if len(url) > 100:
        return RuleMatch(
            rule_id="URL_002",
            severity=Severity.LOW,
            description="Excessive URL Length.",
            evidence={"length": len(url)}
        )
    return None

def check_at_symbol(url, parsed):
    if '@' in parsed.netloc:
        return RuleMatch(
            rule_id="URL_003",
            severity=Severity.HIGH,
            description="Embedded Credentials via @ symbol in domain.",
            evidence={"netloc": parsed.netloc}
        )
    return None

def check_explicit_port(url, parsed):
    if ':' in parsed.netloc and not parsed.netloc.endswith((':80', ':443')):
        return RuleMatch(
            rule_id="URL_004",
            severity=Severity.MEDIUM,
            description="Suspicious explicit port mapping.",
            evidence={"netloc": parsed.netloc}
        )
    return None

def check_excessive_encoding(url, parsed):
    num_encodings = len(re.findall(r'%[0-9a-fA-F]{2}', url))
    if num_encodings > 5:
        return RuleMatch(
            rule_id="URL_005",
            severity=Severity.LOW,
            description="Excessive URL Encoding.",
            evidence={"encoding_count": num_encodings}
        )
    return None

def check_suspicious_patterns(url, parsed):
    suspicious = ['login', 'admin', 'cmd=', 'exec', 'password', 'secure']
    found = [s for s in suspicious if s in url.lower()]
    if found:
        return RuleMatch(
            rule_id="URL_006",
            severity=Severity.MEDIUM,
            description="Suspicious path or query patterns detected.",
            evidence={"patterns": found}
        )
    return None
