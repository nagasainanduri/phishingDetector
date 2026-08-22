import os
import posixpath
import urllib.parse
from dataclasses import dataclass, field
from typing import List
import ipaddress
import re

@dataclass
class CanonicalizationResult:
    raw_url: str
    canonical_url: str
    hostname: str
    normalized_hostname: str
    transformations: List[str] = field(default_factory=list)
    encoding_depth: int = 0
    suspicious_encoding: bool = False


class URLCanonicalizer:
    def __init__(self):
        self.max_depth = int(os.environ.get('PHISHGUARD_MAX_DECODE_DEPTH', '5'))

    def canonicalize(self, url: str) -> CanonicalizationResult:
        original_url = url
        current_url = url
        transformations = []
        depth = 0
        suspicious = False

        # 1. Bounded Recursive Percent Decoding
        while depth < self.max_depth:
            decoded = urllib.parse.unquote(current_url)
            if decoded == current_url:
                break
            current_url = decoded
            depth += 1
            if depth == 1:
                transformations.append("percent_decoded")
        
        if depth >= self.max_depth:
            suspicious = True
            transformations.append("max_decode_depth_reached")

        # Basic cleanup
        if not re.match(r'^[a-zA-Z]+://', current_url):
            current_url = 'http://' + current_url

        try:
            parsed = urllib.parse.urlparse(current_url)
            hostname = parsed.hostname or ""
            normalized_hostname = hostname
            
            # 2. Punycode / IDNA
            if hostname.startswith("xn--") or ".xn--" in hostname:
                try:
                    normalized_hostname = hostname.encode('utf-8').decode('idna')
                    transformations.append("punycode_decoded")
                except Exception:
                    suspicious = True
                    transformations.append("invalid_punycode")

            # 3. Hexadecimal / Integer IP parsing
            ip_match = False
            # Check for hex IP
            if re.match(r'^0x[0-9a-fA-F]+$', hostname):
                try:
                    ip_int = int(hostname, 16)
                    normalized_hostname = str(ipaddress.IPv4Address(ip_int))
                    transformations.append("hex_ip_decoded")
                    suspicious = True
                    ip_match = True
                except Exception:
                    pass
            # Check for integer IP
            elif re.match(r'^\d+$', hostname):
                try:
                    ip_int = int(hostname)
                    normalized_hostname = str(ipaddress.IPv4Address(ip_int))
                    transformations.append("int_ip_decoded")
                    suspicious = True
                    ip_match = True
                except Exception:
                    pass
            
            # 4. Path Normalization
            normalized_path = posixpath.normpath(parsed.path) if parsed.path else ""
            if normalized_path != parsed.path and parsed.path:
                transformations.append("path_normalized")
            
            # Reconstruct
            canonical_url = urllib.parse.urlunparse((
                parsed.scheme,
                normalized_hostname + (f":{parsed.port}" if parsed.port else ""),
                normalized_path,
                parsed.params,
                parsed.query,
                parsed.fragment
            ))

        except Exception as e:
            # Fallback if parsing completely fails
            canonical_url = current_url
            hostname = ""
            normalized_hostname = ""
            suspicious = True
            transformations.append("parsing_failed")

        return CanonicalizationResult(
            raw_url=original_url,
            canonical_url=canonical_url,
            hostname=hostname,
            normalized_hostname=normalized_hostname,
            transformations=transformations,
            encoding_depth=depth,
            suspicious_encoding=suspicious
        )
