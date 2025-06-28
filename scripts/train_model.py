import pandas as pd
import pickle
import time
import os 
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
from feature_extraction import extract_features 
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urljoin

def reconstruct_url(row):
    """Reconstruct URL from dataset3.csv."""
    protocol = row['protocol'].strip() if pd.notna(row['protocol']) else 'http'
    domain = row['domain_name'].strip() if pd.notna(row['domain_name']) else ''
    address = row['address'].strip() if pd.notna(row['address']) else ''
    base_url = f"{protocol}://{domain}" if domain else ''
    full_url = urljoin(base_url, address) if base_url and address else base_url
    return full_url if full_url else None

def load_data(dataset3_csv='data/dataset3.csv'):
    """Load processed_data.csv and dataset3.csv for features."""
    try:
        data = pd.read_csv('data/processed_data.csv')
    except FileNotFoundError:
        print("Error: 'data/processed_data.csv' not found. Run 'python scripts/preprocess_datasets.py' first.")
        exit(1)

    dataset3_features = None
    if os.path.exists(dataset3_csv):
        df = pd.read_csv(dataset3_csv)
        df['url'] = df.apply(reconstruct_url, axis=1)
        df['label'] = df['is_phished'].map({'yes': 1, 'no': 0})
        df = df.rename(columns={
            'long_url': 'url_length',
            'having_@_symbol': 'has_at',
            'prefix_suffix_seperation': 'has_dash',
            'sub_domains': 'has_subdomain'
        })
        dataset3_features = df[['url', 'url_length', 'has_at', 'has_dash', 'has_subdomain']].dropna()
        dataset3_features = dataset3_features.drop_duplicates(subset=['url'], keep='first')
    return data, dataset3_features

def safe_extract(url):
    try:
        return url, extract_features(url)
    except Exception as e:
        print(f"Error extracting features for {url}: {e}")
        return url, None

def extract_features_from_dataset(data, dataset3_features=None):
    print("Extracting features from URLs (parallel)...")
    features = []
    failed_urls = []
    successful_urls = []
    
    # URLs needing feature extraction (not in dataset3)
    urls_to_extract = data['url']
    if dataset3_features is not None:
        urls_to_extract = data[~data['url'].isin(dataset3_features['url'])]['url']
    
    # Parallel feature extraction
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(safe_extract, url): url for url in urls_to_extract}
        for future in tqdm(as_completed(futures), total=len(futures)):
            try:
                url, result = future.result()
                if result:
                    features.append(result)
                    successful_urls.append(url)
                else:
                    failed_urls.append(url)
            except Exception as e:
                failed_urls.append(futures[future])
                print(f"URL failed: {futures[future]} — {e}")
    
    features_df = pd.DataFrame(features)
    features_df['url'] = successful_urls
    
    # Merge dataset3 features (only for URLs in data)
    if dataset3_features is not None:
        dataset3_subset = dataset3_features[dataset3_features['url'].isin(data['url'])][['url', 'url_length', 'has_at', 'has_dash', 'has_subdomain']]
        features_df = pd.concat([features_df, dataset3_subset], ignore_index=True)
    
    # Deduplicate URLs
    original_len = len(features_df)
    features_df = features_df.drop_duplicates(subset=['url'], keep='first')
    if len(features_df) < original_len:
        print(f"Removed {original_len - len(features_df)} duplicate URLs from features_df")
    successful_urls = features_df['url'].tolist()
    
    # Fill missing features with defaults
    for col in ['has_ip', 'https', 'num_dots', 'num_slashes', 'has_query', 'domain_length', 'tld_length', 'dns_record', 'num_anchors', 'external_anchors', 'num_forms', 'has_popup', 'meta_refresh', 'domain_age']:
        if col not in features_df:
            features_df[col] = 0 if col != 'domain_age' else -1
    
    # Log failed URLs
    if failed_urls:
        os.makedirs('logs', exist_ok=True)
        with open('logs/failed_urls.log', 'a') as f:
            for url in failed_urls:
                f.write(f"{url} - Failed at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        print(f"Logged {len(failed_urls)} failed URLs to logs/failed_urls.log")
    
    return features_df.drop(columns=['url']), successful_urls

def train_model():
    start = time.time()
    data, dataset3_features = load_data()
    X, successful_urls = extract_features_from_dataset(data, dataset3_features)
    # Filter data to match successful URLs
    filtered_data = data[data['url'].isin(successful_urls)]
    y = filtered_data['label']

    # Log lengths => intended for debugging
    print(f"X length: {len(X)}, y length: {len(y)}, filtered_data length: {len(filtered_data)}")
    
    if len(X) != len(y):
        print(f"Error: X ({len(X)}) and y ({len(y)}) have inconsistent lengths.")
        mismatched_urls = data[~data['url'].isin(successful_urls)]['url'].tolist()
        extra_urls = [url for url in successful_urls if url not in data['url'].tolist()]
        print(f"Sample mismatched URLs (not in X): {mismatched_urls[:5]}")
        print(f"Sample extra URLs (in X but not y): {extra_urls[:5]}")
        os.makedirs('logs', exist_ok=True)
        with open('logs/mismatched_urls.log', 'a') as f:
            for url in mismatched_urls:
                f.write(f"{url} - Mismatched (not in X) at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            for url in extra_urls:
                f.write(f"{url} - Extra (in X but not y) at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        print("Logged mismatched URLs to logs/mismatched_urls.log")
        exit(1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight='balanced')
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print(f"\nAccuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
    print(f"Precision: {precision_score(y_test, y_pred) * 100:.2f}%")
    print(f"Recall: {recall_score(y_test, y_pred) * 100:.2f}%")

    importances = model.feature_importances_
    feature_names = X.columns
    print("\nFeature Importances:")
    for name, imp in zip(feature_names, importances):
        print(f"{name}: {imp:.4f}")

    os.makedirs('models', exist_ok=True)
    with open('models/phishing_detector.pkl', 'wb') as f:
        pickle.dump(model, f)
    print("Model saved to 'models/phishing_detector.pkl'")
    print(f"Training completed in {round(time.time() - start, 2)} seconds.")

if __name__ == '__main__':
    train_model()