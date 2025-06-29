import pandas as pd
import os
import time
import logging
import re
from urllib.parse import urlparse

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def validate_url(url, kaggele_mode=False):
    """Validate if a string is a proper URL with a domain or IP."""
    if not url or not isinstance(url, str):
        return False
    try:
        #for kaggele dataset => this dataset does not have http:// or https:// in the URL
        if kaggele_mode and not url.startswith(('http://', 'https://')):
            url = f"https://{url}"
        parsed = urlparse(url)
        
        if not kaggele_mode and not parsed.scheme in ['http', 'https']:
            return False
        if not parsed.netloc:
            return False

        if re.match(r'^\d+$', parsed.netloc):
            return False  # Just a number, not a valid domain or IP
        
        # Accept domains or IPs
        if re.match(r'^([a-zA-Z0-9-]+\.)*[a-zA-Z0-9-]+\.[a-zA-Z]{2,}$|^(\d{1,3}\.){3}\d{1,3}$', parsed.netloc):
            return True
        return False
    except Exception:
        return False

def reconstruct_url(df):
    """Vectorized URL reconstruction."""
    try:
        protocol = df['protocol'].fillna('http').str.strip()
        domain = df['domain_name'].fillna('').str.strip()
        address = df['address'].fillna('').str.strip()
        base_url = protocol + '://' + domain
        full_url = base_url + '/' + address
        full_url = full_url.str.replace('//+', '/', regex=True).str.replace(':/', '://', regex=False)
        return full_url.where(full_url != 'http://', None)
    except Exception as e:
        logger.error(f"Error in reconstruct_url: {type(e).__name__} - {str(e)}")
        return pd.Series([None] * len(df), index=df.index)

def preprocess_dataset3(input_csv):
    """Process dataset3.csv into url and label."""
    try:
        df = pd.read_csv(input_csv)
        df['url'] = reconstruct_url(df)
        df['label'] = df['is_phished'].map({'yes': 1, 'no': 0})
        # Validate URLs
        invalid_urls = df['url'][~df['url'].apply(validate_url)].tolist()
        if invalid_urls:
            logger.warning(f"Dropped {len(invalid_urls)} invalid URLs from {input_csv}: {invalid_urls[:5]}")
            with open('logs/invalid_urls.log', 'a') as f:
                for url in invalid_urls[:5]:
                    f.write(f"{url} - Invalid URL at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        df = df[df['url'].apply(validate_url)]
        result = df[['url', 'label']].dropna()
        logger.info(f"Processed {len(result)} URLs from {input_csv}")
        return result
    except Exception as e:
        logger.error(f"Error processing {input_csv}: {type(e).__name__} - {str(e)}")
        return pd.DataFrame(columns=['url', 'label'])

def preprocess_kaggle(input_csv):
    """Process Kaggle dataset into url and label."""
    try:
        df = pd.read_csv(input_csv)
        expected_cols = {'URL', 'LABEL'}
        if not all(col in df.columns for col in expected_cols):
            logger.error(f"Expected columns 'URL' and 'LABEL' in {input_csv}. Found: {list(df.columns)}")
            return pd.DataFrame(columns=['url', 'label'])

        df = df.rename(columns={'URL': 'url', 'LABEL': 'label'})
        logger.info(f"Renamed columns in {input_csv} to 'url' and 'label'")
        original_len = len(df)
        
        #prepend http protols
        df['url'] = df['url'].apply(lambda x: f"https://{x}" if x and isinstance(x, str) and not x.startswith(('http://', 'https://')) else x)
        logger.info(f"Prepended https:// to protocol-less urls in {input_csv}")
        
        # Validate URLs
        invalid_urls = df['url'][~df['url'].apply(validate_url)].tolist()
        if invalid_urls:
            logger.warning(f"Dropped {len(invalid_urls)} invalid URLs from {input_csv}: {invalid_urls[:5]}")
            with open('logs/invalid_urls.log', 'a') as f:
                for url in invalid_urls[:5]:
                    f.write(f"{url} - Invalid URL at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        df = df[df['url'].apply(validate_url)]
        
        # Map labels
        df['label'] = df['label'].map({
            'Good': 0, 'good': 0, 'Legitimate': 0, 'legitimate': 0, 0: 0, '0': 0,
            'Bad': 1, 'bad': 1, 'Phishing': 1, 'phishing': 1, 1: 1, '1': 1
        })
        dropped = original_len - len(df)
        if df['label'].isna().any():
            na_count = df['label'].isna().sum()
            logger.warning(f"Dropped {na_count} rows from {input_csv} due to invalid labels")
            df = df.dropna()
        if dropped > 0:
            logger.info(f"Processed {len(df)} URLs from {input_csv} (dropped {dropped} rows)")
        result = df[['url', 'label']].dropna()
        logger.info(f"Returning {len(result)} valid URLs from {input_csv}")
        return result
    except Exception as e:
        logger.error(f"Error processing {input_csv}: {type(e).__name__} - {str(e)}")
        return pd.DataFrame(columns=['url', 'label'])

def preprocess_alexa(input_csv):
    """Process Alexa Top 1M into url and label."""
    try:
        df = pd.read_csv(input_csv)
        url_col = 'domain' if 'domain' in df.columns else df.columns[0]
        df['url'] = df[url_col].astype(str).apply(lambda x: f"https://{x.strip()}" if pd.notna(x) and x.strip() else None)
        # Validate URLs
        invalid_urls = df['url'][~df['url'].apply(validate_url)].tolist()
        if invalid_urls:
            logger.warning(f"Dropped {len(invalid_urls)} invalid URLs from {input_csv}: {invalid_urls[:5]}")
            with open('logs/invalid_urls.log', 'a') as f:
                for url in invalid_urls[:5]:
                    f.write(f"{url} - Invalid URL at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        df = df[df['url'].apply(validate_url)]
        df['label'] = 0
        result = df[['url', 'label']].dropna()
        logger.info(f"Processed {len(result)} URLs from {input_csv}")
        return result
    except Exception as e:
        logger.error(f"Error processing {input_csv}: {type(e).__name__} - {str(e)}")
        return pd.DataFrame(columns=['url', 'label'])

def process_csv_files(dataset3_csv='data/dataset3.csv', alexa_csv='data/alexa_top1m.csv', kaggle_csv='data/kaggle_phishing.csv'):
    """Combine dataset3.csv, Alexa Top 1M, and Kaggle dataset."""
    os.makedirs('data', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    final_df = pd.DataFrame(columns=['url', 'label'])
    
    if os.path.exists(dataset3_csv):
        dataset3_df = preprocess_dataset3(dataset3_csv)
        final_df = pd.concat([final_df, dataset3_df], ignore_index=True)
        logger.info(f"Processed {len(dataset3_df)} URLs from {dataset3_csv}")
    else:
        logger.warning(f"{dataset3_csv} not found")
    
    if os.path.exists(kaggle_csv):
        kaggle_df = preprocess_kaggle(kaggle_csv)
        final_df = pd.concat([final_df, kaggle_df], ignore_index=True)
        logger.info(f"Processed {len(kaggle_df)} URLs from {kaggle_csv}")
    else:
        logger.warning(f"{kaggle_csv} not found")
    
    if os.path.exists(alexa_csv):
        alexa_df = preprocess_alexa(alexa_csv)
        final_df = pd.concat([final_df, alexa_df], ignore_index=True)
        logger.info(f"Processed {len(alexa_df)} legit URLs from {alexa_csv}")
    else:
        logger.warning(f"{alexa_csv} not found")
    
    original_len = len(final_df)
    duplicate_urls = final_df[final_df.duplicated(subset=['url'], keep=False)]['url'].tolist()
    final_df = final_df.drop_duplicates(subset=['url'], keep='first').dropna()
    dropped = original_len - len(final_df)
    if dropped > 0:
        logger.warning(f"Dropped {dropped} URLs due to duplicates or NaN values")
        with open('logs/dropped_urls.log', 'a') as f:
            for url in duplicate_urls[:5]:
                f.write(f"{url} - Dropped (duplicate) at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            na_urls = final_df[final_df['url'].isna() | final_df['label'].isna()]['url'].tolist()
            for url in na_urls[:5]:
                f.write(f"{url} - Dropped (NaN) at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        logger.info(f"Logged dropped URLs to logs/dropped_urls.log")
    
    phishing_count = len(final_df[final_df['label'] == 1])
    legit_count = len(final_df[final_df['label'] == 0])
    target_count = min(phishing_count, legit_count, 18000)#limit to each kind of data => phishing and legit to 25000 each
    phishing = final_df[final_df['label'] == 1].sample(n=target_count, random_state=42)
    legit = final_df[final_df['label'] == 0].sample(n=target_count, random_state=42)
    final_df = pd.concat([phishing, legit], ignore_index=True)
    logger.info(f"Sampled {len(phishing)} phishing and {len(legit)} legit URLs")
    
    final_df.to_csv('data/processed_data.csv', index=False)
    logger.info(f"Saved {len(final_df)} URLs to data/processed_data.csv (Phishing: {len(final_df[final_df['label'] == 1])}, Legitimate: {len(final_df[final_df['label'] == 0])})")

if __name__ == "__main__":
    process_csv_files()