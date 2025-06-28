import pandas as pd
import os
import time

def reconstruct_url(df):
    """Vectorized URL reconstruction."""
    protocol = df['protocol'].fillna('http').str.strip()
    domain = df['domain_name'].fillna('').str.strip()
    address = df['address'].fillna('').str.strip()
    base_url = protocol + '://' + domain
    full_url = base_url + '/' + address
    full_url = full_url.replace('//', '/').replace(':/', '://')  # To clean up the URL
    return full_url.where(full_url != 'http://', None)

def preprocess_dataset3(input_csv):
    """Process dataset3.csv into url and label."""
    try:
        df = pd.read_csv(input_csv)
        df['url'] = reconstruct_url(df)
        df['label'] = df['is_phished'].map({'yes': 1, 'no': 0})
        return df[['url', 'label']].dropna()
    except Exception as e:
        print(f"Error processing {input_csv}: {str(e)}")
        return pd.DataFrame(columns=['url', 'label'])

def preprocess_kaggle(input_csv):
    """Process Kaggle dataset into url and label."""
    try:
        df = pd.read_csv(input_csv)
        # Find URL and Label columns (case-insensitive)
        url_col = next((col for col in df.columns if col.lower() in ['url', 'website', 'address']), None)
        label_col = next((col for col in df.columns if col.lower() in ['label', 'result', 'target', 'class']), None)
        
        if not url_col or not label_col:
            print(f"Error: {input_csv} missing URL or Label column. Found: {list(df.columns)}")
            return pd.DataFrame(columns=['url', 'label'])
        
        df = df.rename(columns={url_col: 'url', label_col: 'label'})
        # To Handle common label variations
        original_len = len(df)
        df['label'] = df['label'].map({
            'Good': 0, 'good': 0, 'Legitimate': 0, 'legitimate': 0, 0: 0, '0': 0,
            'Bad': 1, 'bad': 1, 'Phishing': 1, 'phishing': 1, 1: 1, '1': 1
        })
        dropped = original_len - len(df)
        if df['label'].isna().any():
            na_count = df['label'].isna().sum()
            print(f"Warning: Dropped {na_count} rows from {input_csv} due to invalid label values: {df[df['label'].isna()]['url'].head().tolist()}")
            df = df.dropna()
        if dropped > 0:
            print(f"Processed {len(df)} URLs from {input_csv} (dropped {dropped} rows)")
        return df[['url', 'label']].dropna()
    except Exception as e:
        print(f"Error processing {input_csv}: {str(e)}")
        return pd.DataFrame(columns=['url', 'label'])

def preprocess_alexa(input_csv):
    """Process Alexa Top 1M into url and label."""
    try:
        df = pd.read_csv(input_csv)
        url_col = 'domain' if 'domain' in df.columns else df.columns[0]
        # Convert to string to handle integers
        df['url'] = df[url_col].astype(str).apply(lambda x: f"https://{x.strip()}" if pd.notna(x) and x.strip() else None)
        df['label'] = 0
        return df[['url', 'label']].dropna()
    except Exception as e:
        print(f"Error processing {input_csv}: {str(e)}")
        return pd.DataFrame(columns=['url', 'label'])

def process_csv_files(dataset3_csv='data/dataset3.csv', alexa_csv='data/alexa_top1m.csv', kaggle_csv='data/kaggle_phishing.csv'):
    """Combine dataset3.csv, Alexa Top 1M, and Kaggle dataset."""
    final_df = pd.DataFrame(columns=['url', 'label'])
    
    # Process dataset3.csv
    if os.path.exists(dataset3_csv):
        dataset3_df = preprocess_dataset3(dataset3_csv)
        final_df = pd.concat([final_df, dataset3_df], ignore_index=True)
        print(f"Processed {len(dataset3_df)} URLs from {dataset3_csv}")
    else:
        print(f"Warning: {dataset3_csv} not found")
    
    # Process Kaggle dataset
    if os.path.exists(kaggle_csv):
        kaggle_df = preprocess_kaggle(kaggle_csv)
        final_df = pd.concat([final_df, kaggle_df], ignore_index=True)
        print(f"Processed {len(kaggle_df)} URLs from {kaggle_csv}")
    else:
        print(f"Warning: {kaggle_csv} not found")
    
    # Process Alexa Top 1M
    if os.path.exists(alexa_csv):
        alexa_df = preprocess_alexa(alexa_csv)
        final_df = pd.concat([final_df, alexa_df], ignore_index=True)
        print(f"Processed {len(alexa_df)} legit URLs from {alexa_csv}")
    else:
        print(f"Warning: {alexa_csv} not found")
    
    # Remove duplicates and NaN
    original_len = len(final_df)
    duplicate_urls = final_df[final_df.duplicated(subset=['url'], keep=False)]['url'].tolist()
    final_df = final_df.drop_duplicates(subset=['url'], keep='first').dropna()
    dropped = original_len - len(final_df)
    if dropped > 0:
        print(f"Warning: Dropped {dropped} URLs due to duplicates or NaN values")
        os.makedirs('logs', exist_ok=True)
        with open('logs/dropped_urls.log', 'a') as f:
            for url in duplicate_urls[:5]:
                f.write(f"{url} - Dropped (duplicate) at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            na_urls = final_df[final_df['url'].isna() | final_df['label'].isna()]['url'].tolist()
            for url in na_urls[:5]:
                f.write(f"{url} - Dropped (NaN) at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        print(f"Logged dropped URLs to logs/dropped_urls.log")
    
    # Sample balanced dataset (~18,000 each)
    phishing_count = len(final_df[final_df['label'] == 1])
    legit_count = len(final_df[final_df['label'] == 0])
    target_count = min(phishing_count, legit_count, 18000)  # Target ~18K each
    phishing = final_df[final_df['label'] == 1].sample(n=target_count, random_state=42)
    legit = final_df[final_df['label'] == 0].sample(n=target_count, random_state=42)
    final_df = pd.concat([phishing, legit], ignore_index=True)
    print(f"Sampled {len(phishing)} phishing and {len(legit)} legit URLs")
    
    # Save to processed_data.csv
    os.makedirs('data', exist_ok=True)
    final_df.to_csv('data/processed_data.csv', index=False)
    print(f"Saved {len(final_df)} URLs to data/processed_data.csv (Phishing: {len(final_df[final_df['label'] == 1])}, Legitimate: {len(final_df[final_df['label'] == 0])})")

if __name__ == "__main__":
    process_csv_files()