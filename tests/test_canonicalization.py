import pytest
from detector.canonicalization import URLCanonicalizer

def test_canonicalize_clean_url():
    c = URLCanonicalizer()
    res = c.canonicalize("http://example.com/path")
    assert res.canonical_url == "http://example.com/path"
    assert res.hostname == "example.com"
    assert res.encoding_depth == 0
    assert not res.suspicious_encoding

def test_canonicalize_hex_ip():
    c = URLCanonicalizer()
    res = c.canonicalize("http://0x7F000001/login")
    assert res.normalized_hostname == "127.0.0.1"
    assert "hex_ip_decoded" in res.transformations
    assert res.suspicious_encoding

def test_canonicalize_int_ip():
    c = URLCanonicalizer()
    res = c.canonicalize("http://2130706433/login")
    assert res.normalized_hostname == "127.0.0.1"
    assert "int_ip_decoded" in res.transformations
    assert res.suspicious_encoding

def test_canonicalize_punycode():
    c = URLCanonicalizer()
    res = c.canonicalize("http://xn--bcher-kva.example.com")
    assert "punycode_decoded" in res.transformations

def test_canonicalize_percent_encoding():
    c = URLCanonicalizer()
    res = c.canonicalize("http://example.com/%70%61%74%68")
    assert res.canonical_url == "http://example.com/path"
    assert "percent_decoded" in res.transformations
    assert res.encoding_depth == 1

def test_canonicalize_recursive_encoding():
    c = URLCanonicalizer()
    # Path is deeply encoded: %2525252520 -> %25252520 -> %252520 -> %2520 -> %20 -> " "
    res = c.canonicalize("http://example.com/%2525252520")
    assert res.encoding_depth > 1
    assert "percent_decoded" in res.transformations

def test_canonicalize_max_depth():
    import os
    os.environ['PHISHGUARD_MAX_DECODE_DEPTH'] = '2'
    c = URLCanonicalizer()
    # %2525252520 needs 5 decodings. Max is 2.
    res = c.canonicalize("http://example.com/%2525252520")
    assert res.encoding_depth == 2
    assert "max_decode_depth_reached" in res.transformations
    assert res.suspicious_encoding
