from feature_extraction import extract_features

test_urls = [
    "mariachisadomicilio.cl/images/ben/0891d882490682133e446684b3618aba/webscr.php?cmd=_login-run&dispatch=5885d80a13c0db1f1ff80d546411d7f8a8350c132bc41e0934cfc023d4e8f9e5",
    "ekonova.nazwa.pl/wc0coj",
    "www.paypal.com.4vll44l52csp.0488wuf2jqytv4n1.com/cgi-bin/webscr/?login-dispatch&login_email=nshost.com.ve@att.net&ref=pp&login-processing=ok",
    "waince.com/wawa/online.wellsfargo.com/online.wellsfargo.com/security-update/youraccounts/"
    "http://example.com",
    "https://google.com",
    "12345"
]
for url in test_urls:
    result = extract_features(url)
    print(f"{url}: {result}")