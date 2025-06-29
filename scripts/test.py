import pickle
with open('models/feature_cache.pkl', 'rb') as f:
    cache = pickle.load(f)
print(f"Cache entries: {len(cache)}")
print(list(cache.items())[:5])