from .scoring import RuleMatch, Severity

def check_credential_collection(url, parsed, page_signals=None):
    if not page_signals:
        return None
        
    if page_signals.get('has_password_field') and parsed.scheme != 'https':
        return RuleMatch(
            rule_id="PAGE_001",
            severity=Severity.HIGH,
            description="Page collects passwords but is not served over HTTPS.",
            evidence={"has_password_field": True, "scheme": parsed.scheme}
        )
    return None

def check_cross_origin_action(url, parsed, page_signals=None):
    if not page_signals:
        return None
        
    if page_signals.get('cross_origin_form_action'):
        # If it also has a password field, it's highly suspicious
        if page_signals.get('has_password_field') or page_signals.get('has_login_form'):
            return RuleMatch(
                rule_id="PAGE_002",
                severity=Severity.CRITICAL,
                description="Login/Password form submits data to a different domain.",
                evidence={"cross_origin_action": True, "has_login": True}
            )
        else:
            return RuleMatch(
                rule_id="PAGE_002",
                severity=Severity.MEDIUM,
                description="Form submits data to a different domain.",
                evidence={"cross_origin_action": True, "has_login": False}
            )
    return None

def check_hidden_iframes(url, parsed, page_signals=None):
    if not page_signals:
        return None
        
    if page_signals.get('hidden_iframes'):
        return RuleMatch(
            rule_id="PAGE_003",
            severity=Severity.MEDIUM,
            description="Page contains hidden iframes, commonly used for malicious redirects or invisible rendering.",
            evidence={"hidden_iframes": True}
        )
    return None
