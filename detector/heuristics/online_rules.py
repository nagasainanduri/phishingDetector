import ssl
import socket
import datetime
from .scoring import RuleMatch, Severity

# Known high-reputation Certificate Authorities
TRUSTED_ISSUERS = {
    'Google Trust Services', 'DigiCert', 'GlobalSign', 'Entrust', 
    'Amazon', 'Microsoft', 'Sectigo', 'Symantec', 'Thawte', 'GeoTrust'
}

def check_ssl_certificate(url, parsed):
    """
    Connects to port 443 to verify the SSL certificate.
    Returns HIGH risk if invalid, SAFE if highly reputable.
    """
    if parsed.scheme != 'https':
        return None
        
    if not parsed.hostname:
        return None

    hostname = parsed.hostname.lower()
    ctx = ssl.create_default_context()
    
    try:
        with socket.create_connection((hostname, 443), timeout=2.0) as sock:
            with ctx.wrap_socket(sock, server_hostname=hostname) as ssock:
                cert = ssock.getpeercert()
                
                # Check expiration
                try:
                    not_after = datetime.datetime.strptime(cert['notAfter'], '%b %d %H:%M:%S %Y %Z')
                    if not_after < datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None):
                        return RuleMatch(
                            rule_id="NET_001",
                            severity=Severity.HIGH,
                            description="SSL Certificate is expired.",
                            evidence={"expired_on": str(not_after)}
                        )
                except Exception:
                    pass
                
                # Check Issuer Reputation
                issuer_dict = dict(x[0] for x in cert.get('issuer', []))
                issuer_org = issuer_dict.get('organizationName', '')
                
                if any(trusted in issuer_org for trusted in TRUSTED_ISSUERS):
                    return RuleMatch(
                        rule_id="NET_002",
                        severity=Severity.SAFE,
                        description="SSL Certificate issued by a highly reputable CA.",
                        evidence={"issuer": issuer_org}
                    )
                    
                return None
                
    except ssl.SSLCertVerificationError as e:
        return RuleMatch(
            rule_id="NET_001",
            severity=Severity.HIGH,
            description="SSL Certificate verification failed (invalid, self-signed, or mismatched hostname).",
            evidence={"error": str(e)}
        )
    except (socket.timeout, ConnectionRefusedError, socket.gaierror):
        # We don't necessarily penalize heavily for a connection timeout on heuristic level,
        # because the server could just be down or blocking our IP.
        return RuleMatch(
            rule_id="NET_003",
            severity=Severity.LOW,
            description="Could not establish SSL connection to the host.",
            evidence={"hostname": hostname}
        )
    except Exception as e:
        return None

def check_threat_intel(url, parsed, threat_intel_findings=None):
    if not threat_intel_findings:
        return None
        
    malicious_providers = []
    for finding in threat_intel_findings:
        if finding.get("status") == "MALICIOUS":
            malicious_providers.append(finding["provider"])
            
    if malicious_providers:
        return RuleMatch(
            rule_id="ONL_002",
            severity=Severity.CRITICAL,
            description=f"Flagged as MALICIOUS by Threat Intelligence ({', '.join(malicious_providers)}).",
            evidence={"threat_intel": threat_intel_findings}
        )
    return None
