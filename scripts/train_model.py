import pandas as pd
import pickle, time, os, psutil, logging
import dask.dataframe as dd
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
from feature_extraction import extract_features, load_cache, save_cache
from concurrent.futures import ThreadPoolExecutor, as_completed

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_data(dataset3_csv='data/dataset3.csv'):
    """Load processed_data.csv and optionally dataset3.csv for precomputed features using Dask."""
    try:
        data = dd.read_csv('data/processed_data.csv').compute()
        logger.info(f"Loaded {len(data)} URLs from data/processed_data.csv")
    except FileNotFoundError:
        logger.error("data/processed_data.csv not found. Run preprocess_datasets.py first.")
        exit(1)

    dataset3_features = None
    if os.path.exists(dataset3_csv):
        try:
            df = dd.read_csv(dataset3_csv).compute()
            df = df.rename(columns={
                'long_url': 'url_length',
                'having_@_symbol': 'has_at',
                'prefix_suffix_seperation': 'has_dash',
                'sub_domains': 'has_subdomain'
            })
            if 'url' in df.columns:
                dataset3_features = df[['url', 'url_length', 'has_at', 'has_dash', 'has_subdomain']].dropna()
                dataset3_features = dataset3_features.drop_duplicates(subset=['url'], keep='first')
                logger.info(f"Loaded {len(dataset3_features)} precomputed features from {dataset3_csv}")
            else:
                logger.warning(f"No 'url' column in {dataset3_csv}. Ignoring precomputed features.")
        except Exception as e:
            logger.error(f"Error loading {dataset3_csv}: {type(e).__name__} - {str(e)}")
    else:
        logger.warning(f"{dataset3_csv} not found. Proceeding without precomputed features.")

    return data, dataset3_features

def safe_extract(url, batch_cache):
    """Safely extract features for a URL, using batch_cache."""
    start_time = time.time()
    try:
        result = extract_features(url, batch_cache=batch_cache)
        logger.debug(f"Extracted features for {url} in {time.time() - start_time:.3f} seconds")
        return url, result
    except Exception as e:
        logger.error(f"Error extracting features for {url}: {type(e).__name__} - {str(e)}")
        return url, None

def extract_features_from_dataset(data, dataset3_features=None, batch_size=500):
    """Extract features from dataset URLs in parallel with batch processing."""
    logger.info("Extracting URL-based features (parallel, no network requests)...")
    features = []
    failed_urls = []
    successful_urls = []
    cache = load_cache()
    cache_hits = 0
    
    urls_to_extract = data['url'].values
    if dataset3_features is not None:
        urls_to_extract = data[~data['url'].isin(dataset3_features['url'])]['url'].values
    
    # Adaptive max_workers
    max_workers = min(12, max(1, int(12 * (1 - psutil.cpu_percent(interval=0.1) / 100))))
    logger.info(f"Using {max_workers} workers based on CPU load")

    # Process URLs in batches
    for i in range(0, len(urls_to_extract), batch_size):
        batch_urls = urls_to_extract[i:i + batch_size]
        batch_cache = {}
        logger.info(f"Processing batch {i//batch_size + 1} with {len(batch_urls)} URLs")
        batch_start = time.time()

        # Check cache hits
        batch_cache_hits = sum(1 for url in batch_urls if url in cache)
        cache_hits += batch_cache_hits
        logger.info(f"Batch {i//batch_size + 1} cache hits: {batch_cache_hits}/{len(batch_urls)} ({batch_cache_hits/len(batch_urls)*100:.2f}%)")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(safe_extract, url, batch_cache): url for url in batch_urls}
            for future in tqdm(as_completed(futures), total=len(futures), desc=f"Batch {i//batch_size + 1}"):
                time.sleep(0.1)  # Minimal delay for thread safety
                try:
                    url, result = future.result()
                    if result:
                        features.append(result)
                        successful_urls.append(url)
                    else:
                        failed_urls.append(url)
                except Exception as e:
                    failed_urls.append(futures[future])
                    logger.error(f"URL failed: {futures[future]} - {type(e).__name__} - {str(e)}")
        
        # Update global cache
        if batch_cache:
            cache.update(batch_cache)
            save_cache(cache)
        
        batch_time = time.time() - batch_start
        logger.info(f"Batch {i//batch_size + 1} completed in {round(batch_time, 2)} seconds ({batch_time/len(batch_urls):.3f} seconds/URL), Memory usage: {psutil.virtual_memory().percent}%")

    logger.info(f"Total cache hits: {cache_hits}/{len(urls_to_extract)} ({cache_hits/len(urls_to_extract)*100:.2f}%)")
    
    features_df = pd.DataFrame(features)
    features_df['url'] = successful_urls
    
    if dataset3_features is not None:
        dataset3_subset = dataset3_features[dataset3_features['url'].isin(data['url'])][['url', 'url_length', 'has_at', 'has_dash', 'has_subdomain']]
        features_df = pd.concat([features_df, dataset3_subset], ignore_index=True)
    
    original_len = len(features_df)
    features_df = features_df.drop_duplicates(subset=['url'], keep='first')
    if len(features_df) < original_len:
        logger.info(f"Removed {original_len - len(features_df)} duplicate URLs from features_df")
    successful_urls = features_df['url'].tolist()
    
    # Feature list (no network-based features)
    for col in ['has_ip', 'https', 'num_dots', 'num_slashes', 'has_query', 'domain_length', 'tld_length', 'dns_record', 'has_at', 'has_dash', 'has_subdomain']:
        if col not in features_df:
            features_df[col] = 0
    
    if failed_urls:
        os.makedirs('logs', exist_ok=True)
        with open('logs/failed_urls.log', 'a') as f:
            for url in failed_urls:
                f.write(f"{url} - Failed at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        logger.info(f"Logged {len(failed_urls)} failed URLs to logs/failed_urls.log")
    
    return features_df.drop(columns=['url']), successful_urls

def train_model():
    """Train the phishing detection model."""
    start = time.time()
    data, dataset3_features = load_data()
    X, successful_urls = extract_features_from_dataset(data, dataset3_features)
    filtered_data = data[data['url'].isin(successful_urls)]
    y = filtered_data['label']

    if len(X) != len(y):
        logger.warning(f"X ({len(X)}) and y ({len(y)}) have inconsistent lengths. Filtering mismatched URLs.")
        valid_urls = set(successful_urls).intersection(data['url'].tolist())
        filtered_data = data[data['url'].isin(valid_urls)]
        X = X[filtered_data['url'].isin(valid_urls)]
        y = filtered_data['label']
        logger.info(f"Proceeding with {len(X)} valid URLs")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1, class_weight='balanced')
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    logger.info(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
    logger.info(f"Precision: {precision_score(y_test, y_pred) * 100:.2f}%")
    logger.info(f"Recall: {recall_score(y_test, y_pred) * 100:.2f}%")

    importances = model.feature_importances_
    feature_names = X.columns
    logger.info("\nFeature Importances:")
    for name, imp in zip(feature_names, importances):
        logger.info(f"{name}: {imp:.4f}")

    os.makedirs('models', exist_ok=True)
    with open('models/phishing_detector.pkl', 'wb') as f:
        pickle.dump(model, f)
    logger.info("Model saved to models/phishing_detector.pkl")
    logger.info(f"Training completed in {round(time.time() - start, 2)} seconds.")

def predict_urls(urls, model_path='models/phishing_detector.pkl', cache_path='models/feature_cache.pkl'):
    """Predict phishing labels for a list of URLs using the trained model and feature cache."""
    try:
        # Load the trained model
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        logger.info(f"Loaded model from {model_path}")

        # Load the feature cache
        cache = load_cache() if os.path.exists(cache_path) else {}
        logger.info(f"Loaded cache from {cache_path} with {len(cache)} entries")

        features = []
        failed_urls = []
        successful_urls = []
        cache_hits = 0
        batch_cache = {}

        # Adaptive max_workers
        max_workers = min(12, max(1, int(12 * (1 - psutil.cpu_percent(interval=0.1) / 100))))
        logger.info(f"Using {max_workers} workers for prediction based on CPU load")

        # Process URLs in batches
        batch_size = 500
        for i in range(0, len(urls), batch_size):
            batch_urls = urls[i:i + batch_size]
            logger.info(f"Processing prediction batch {i//batch_size + 1} with {len(batch_urls)} URLs")
            batch_start = time.time()

            batch_cache_hits = sum(1 for url in batch_urls if url in cache)
            cache_hits += batch_cache_hits
            logger.info(f"Prediction batch {i//batch_size + 1} cache hits: {batch_cache_hits}/{len(batch_urls)} ({batch_cache_hits/len(batch_urls)*100:.2f}%)")

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(safe_extract, url, batch_cache): url for url in batch_urls}
                for future in tqdm(as_completed(futures), total=len(futures), desc=f"Prediction Batch {i//batch_size + 1}"):
                    time.sleep(0.1)  # Minimal delay
                    try:
                        url, result = future.result()
                        if result:
                            features.append(result)
                            successful_urls.append(url)
                        else:
                            failed_urls.append(url)
                    except Exception as e:
                        failed_urls.append(futures[future])
                        logger.error(f"Prediction URL failed: {futures[future]} - {type(e).__name__} - {str(e)}")
            
            # Update global cache
            if batch_cache:
                cache.update(batch_cache)
                save_cache(cache)
            
            batch_time = time.time() - batch_start
            logger.info(f"Prediction batch {i//batch_size + 1} completed in {round(batch_time, 2)} seconds ({batch_time/len(batch_urls):.3f} seconds/URL), Memory usage: {psutil.virtual_memory().percent}%")

        logger.info(f"Total cache hits: {cache_hits}/{len(urls)} ({cache_hits/len(urls)*100:.2f}%)")

        if not features:
            logger.error("No features extracted for prediction URLs.")
            return None

        features_df = pd.DataFrame(features)
        features_df['url'] = successful_urls

        # Feature list
        for col in ['has_ip', 'https', 'num_dots', 'num_slashes', 'has_query', 'domain_length', 'tld_length', 'dns_record', 'has_at', 'has_dash', 'has_subdomain']:
            if col not in features_df:
                features_df[col] = 0

        X = features_df.drop(columns=['url'])
        predictions = model.predict(X)
        results = pd.DataFrame({
            'url': successful_urls,
            'prediction': ['phishing' if pred == 1 else 'legitimate' for pred in predictions]
        })

        if failed_urls:
            logger.warning(f"Failed to extract features for {len(failed_urls)} URLs: {failed_urls[:5]}")
            with open('logs/failed_prediction_urls.log', 'a') as f:
                for url in failed_urls:
                    f.write(f"{url} - Failed at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

        logger.info(f"Predicted labels for {len(results)} URLs")
        return results

    except Exception as e:
        logger.error(f"Error during prediction: {type(e).__name__} - {str(e)}")
        return None

def log_prediction_results(results, log_path='logs/prediction_results.log'):
    """Log prediction results to a file."""
    try:
        if results is None or results.empty:
            logger.warning("No prediction results to log.")
            return

        os.makedirs('logs', exist_ok=True)
        with open(log_path, 'a') as f:
            for _, row in results.iterrows():
                f.write(f"{row['url']} - {row['prediction']} - {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        logger.info(f"Logged {len(results)} prediction results to {log_path}")
    except Exception as e:
        logger.error(f"Error logging predictions to {log_path}: {type(e).__name__} - {str(e)}")

if __name__ == '__main__':
    # Train the model
    train_model()

    # Predict on example URLs and log results
    test_urls = [
        "https://example.com",
        "https://mariachisadomicilio.cl/images/ben/0891d882490682133e446684b3618aba/webscr.php",
        "https://ekonova.nazwa.pl/wc0coj"
    ]
    predictions = predict_urls(test_urls)
    if predictions is not None:
        print(predictions)
        log_prediction_results(predictions)