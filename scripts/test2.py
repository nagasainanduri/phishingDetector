from preprocess_datasets import validate_url

test_urls = [
    "12345",  # Invalid: no scheme
    "http://example.com",  # Valid
    "https://192.168.1.1",  # Valid IP
    "http://abc",  # Invalid: incomplete domain
    "http://example.com/path",  # Valid
]
for url in test_urls:
    print(f"{url}: {validate_url(url)}")