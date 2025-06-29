from feature_extraction import extract_features, save_cache, load_cache
import pandas as pd
cache = load_cache()
common_urls = pd.read_csv('data/alexa_top1m.csv')['url'].head(1000).apply(lambda x: f"https://{x}").tolist()
batch_cache = {}
for url in common_urls:
    if url not in cache:
        extract_features(url, batch_cache=batch_cache)
cache.update(batch_cache)
save_cache(cache)