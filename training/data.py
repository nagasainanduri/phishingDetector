# training/data.py
import pandas as pd
import os
import time
import psutil
import logging
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from urllib.parse import urlparse

from detector.features.url_features import extract_features, load_cache, save_cache

logger = logging.getLogger(__name__)

def safe_extract(url, batch_cache):
    """Safely extract features for a URL, using batch_cache."""
    try:
        result = extract_features(url, batch_cache=batch_cache)
        return url, result
    except Exception as e:
        logger.error(f"Error extracting features for {url}: {e}")
        return url, None

class DataLoader:
    def __init__(self, data_path='data/processed_data.csv', precomputed_path='data/dataset3.csv'):
        self.data_path = data_path
        self.precomputed_path = precomputed_path

    def load_raw_data(self):
        """Loads the raw processed_data and optionally dataset3 for precomputed features."""
        try:
            data = pd.read_csv(self.data_path)
            logger.info(f"Loaded {len(data)} URLs from {self.data_path}")
        except FileNotFoundError:
            logger.error(f"{self.data_path} not found.")
            raise

        dataset3_features = None
        if os.path.exists(self.precomputed_path):
            try:
                df = pd.read_csv(self.precomputed_path)
                df = df.rename(columns={
                    'long_url': 'url_length',
                    'having_@_symbol': 'has_at',
                    'prefix_suffix_seperation': 'has_dash',
                    'sub_domains': 'has_subdomain'
                })

                if 'url' in df.columns:
                    dataset3_features = df[['url', 'url_length', 'has_at', 'has_dash', 'has_subdomain']].dropna()
                    dataset3_features = dataset3_features.drop_duplicates(subset=['url'], keep='first')
                    logger.info(f"Loaded {len(dataset3_features)} precomputed features from {self.precomputed_path}")
            except Exception as e:
                logger.error(f"Error loading {self.precomputed_path}: {e}")
                
        return data, dataset3_features

    def get_features(self, batch_size=500):
        """Builds or returns the feature dataframe X and labels y."""
        data, dataset3_features = self.load_raw_data()
        
        urls_to_extract = data['url'].values
        if dataset3_features is not None:
            urls_to_extract = data[~data['url'].isin(dataset3_features['url'])]['url'].values
            
        cache = load_cache()
        features = []
        successful_urls = []
        
        max_workers = min(12, max(1, int(12 * (1 - psutil.cpu_percent(interval=0.1) / 100))))
        logger.info(f"Extracting features using {max_workers} workers")
        
        for i in range(0, len(urls_to_extract), batch_size):
            batch_urls = urls_to_extract[i:i + batch_size]
            batch_cache = {}
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(safe_extract, url, batch_cache): url for url in batch_urls}
                for future in tqdm(as_completed(futures), total=len(futures), desc=f"Batch {i//batch_size + 1}"):
                    try:
                        url, result = future.result()
                        if result:
                            features.append(result)
                            successful_urls.append(url)
                    except Exception as e:
                        pass
                        
            if batch_cache:
                cache.update(batch_cache)
                save_cache(cache)
                
        features_df = pd.DataFrame(features)
        if len(features_df) > 0:
            features_df['url'] = successful_urls
        else:
            features_df = pd.DataFrame(columns=['url'])
            
        if dataset3_features is not None:
            dataset3_subset = dataset3_features[dataset3_features['url'].isin(data['url'])][['url', 'url_length', 'has_at', 'has_dash', 'has_subdomain']]
            features_df = pd.concat([features_df, dataset3_subset], ignore_index=True)
            
        features_df = features_df.drop_duplicates(subset=['url'], keep='first')
        
        expected_cols = ['has_ip', 'https', 'num_dots', 'num_slashes', 'has_query', 'domain_length', 'tld_length', 'dns_record', 'has_at', 'has_dash', 'has_subdomain', 'url_length']
        for col in expected_cols:
            if col not in features_df:
                features_df[col] = 0
                
        merged = features_df.merge(data[['url', 'label']], on='url', how='inner')
        
        y = merged['label']
        urls = merged['url']
        X = merged.drop(columns=['url', 'label'])
        
        return X, y, urls

class DatasetSplitter:
    @staticmethod
    def _get_domain(url):
        try:
            return urlparse(url).netloc
        except:
            return url

    @staticmethod
    def split(X, y, urls, test_size=0.2, random_state=42, domain_aware=True):
        if not domain_aware:
            return train_test_split(X, y, test_size=test_size, random_state=random_state)
            
        domains = urls.apply(DatasetSplitter._get_domain)
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        train_idx, test_idx = next(gss.split(X, y, groups=domains))
        
        return X.iloc[train_idx], X.iloc[test_idx], y.iloc[train_idx], y.iloc[test_idx]
