import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import csv
import os
import time
from datetime import datetime
from scripts.feature_extraction import extract_features
import pickle

SEED_SITES = [
    "https://www.wikipedia.org",
    "https://www.reddit.com",
    "https://www.bbc.com",
    "https://www.cnn.com"
]

CRAWL_LIMIT = 50
DATASET_CSV = "../data/auto_dataset.csv"
LOG_FILE = "../logs/crawler.log"
REPEAT_DOMAIN_LOG = "../logs/repeated_domains.log"
UNREACHABLE_LOG = "../logs/unreachable_domains.log"

visited_domains = set()

def log_message(msg, log_path):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, 'a') as f:
        f.write(f"[{datetime.now()}] {msg}\n")

def is_reachable(url):
    try:
        response = requests.head(url, timeout=5)
        return response.status_code < 400
    except:
        return False

def extract_links(base_url):
    try:
        response = requests.get(base_url, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        links = set()
        for a_tag in soup.find_all("a", href=True):
            href = a_tag['href']
            full_url = urljoin(base_url, href)
            parsed = urlparse(full_url)
            if parsed.scheme.startswith('http'):
                links.add(full_url)
        return list(links)
    except Exception as e:
        log_message(f"Failed to extract links from {base_url}: {e}", UNREACHABLE_LOG)
        return []

def crawl():
    collected = 0
    with open(DATASET_CSV, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if os.stat(DATASET_CSV).st_size == 0:
            writer.writerow(['url', 'label'])

        for seed in SEED_SITES:
            links = extract_links(seed)
            for url in links:
                if collected >= CRAWL_LIMIT:
                    return
                domain = urlparse(url).netloc
                if domain in visited_domains:
                    log_message(f"Repeated domain: {domain}", REPEAT_DOMAIN_LOG)
                    continue
                visited_domains.add(domain)

                if not is_reachable(url):
                    log_message(f"Unreachable domain: {url}", UNREACHABLE_LOG)
                    continue

                try:
                    features = extract_features(url)
                    with open('models/phishing_detector.pkl', 'rb') as f:
                        model = pickle.load(f)
                    label = model.predict([list(features.values())])[0]

                    writer.writerow([url, label])
                    log_message(f"Added: {url} - Label: {label}", LOG_FILE)
                    collected += 1
                except Exception as e:
                    log_message(f"Error processing {url}: {e}", UNREACHABLE_LOG)

if __name__ == "__main__":
    crawl()
