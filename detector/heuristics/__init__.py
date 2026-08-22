from .scoring import Severity, RuleMatch, HeuristicEngine
from .url_rules import (check_ip_address, check_excessive_length, check_at_symbol, 
                        check_explicit_port, check_excessive_encoding, check_suspicious_patterns)
from .domain_rules import (check_excessive_subdomains, check_punycode, 
                           check_url_shortener, check_typosquatting)

def create_engine() -> HeuristicEngine:
    engine = HeuristicEngine()
    engine.register_rule(check_ip_address)
    engine.register_rule(check_excessive_length)
    engine.register_rule(check_at_symbol)
    engine.register_rule(check_explicit_port)
    engine.register_rule(check_excessive_encoding)
    engine.register_rule(check_suspicious_patterns)
    engine.register_rule(check_excessive_subdomains)
    engine.register_rule(check_punycode)
    engine.register_rule(check_url_shortener)
    engine.register_rule(check_typosquatting)
    return engine
