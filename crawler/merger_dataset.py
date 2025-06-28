import pandas as pd
import os
import argparse
from datetime import datetime

PROCESSED_PATH = "../data/processed_data.csv"
AUTO_PATH = "../data/auto_dataset.csv"
LOG_PATH = "../logs/merge_log.log"

def merge_datasets(dry_run=False):
    os.makedirs("data", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    try:
        processed = pd.read_csv(PROCESSED_PATH)
    except FileNotFoundError:
        processed = pd.DataFrame(columns=["url", "label"])

    try:
        auto = pd.read_csv(AUTO_PATH)
    except FileNotFoundError:
        print(f"[{datetime.now()}]No auto_dataset.csv found.")
        return

    before_merge = len(processed)
    combined = pd.concat([processed, auto], ignore_index=True)
    combined.drop_duplicates(subset=["url"], inplace=True)
    new_entries = len(combined) - before_merge
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    if dry_run:
        print(f"Dry Run: {new_entries} new URLs would be added.")
        print(f"Final dataset would contain {len(combined)} total entries.")
        return

    combined.to_csv(PROCESSED_PATH, index=False)
    with open(LOG_PATH, "a") as f:
        f.write(f"[{timestamp}] ✅ Added {new_entries} new URLs from auto_dataset.csv\n")

    print(f"✅ Merge complete — {new_entries} new URLs added.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge crawled data with processed_data.csv")
    parser.add_argument("--dry-run", action="store_true", help="Preview merge without saving changes")
    args = parser.parse_args()

    merge_datasets(dry_run=args.dry_run)
