import pandas as pd
import os
import argparse
from datetime import datetime

FEEDBACK_FILE = 'data/feedback.csv'
DATASET_FILE = 'data/training_urls.csv'

def review_feedback():
    if not os.path.exists(FEEDBACK_FILE):
        print(f"No feedback file found at {FEEDBACK_FILE}.")
        return

    try:
        df_feedback = pd.read_csv(FEEDBACK_FILE)
    except pd.errors.EmptyDataError:
        print("Feedback file is empty.")
        return

    # Filter out hashed URLs because they cannot be validated manually
    # and "correct" predictions since they don't add new signal for failure cases
    actionable_df = df_feedback[
        (~df_feedback['url_identifier'].str.startswith('HASHED:')) &
        (df_feedback['feedback_type'] != 'correct')
    ]

    if actionable_df.empty:
        print("No actionable (raw) false-positive or false-negative feedback to review.")
        print(f"Total feedback entries: {len(df_feedback)}")
        return

    print(f"Found {len(actionable_df)} actionable feedback entries requiring review.")
    
    accepted_entries = []
    
    for idx, row in actionable_df.iterrows():
        print("-" * 50)
        print(f"URL:          {row['url_identifier']}")
        print(f"Feedback:     {row['feedback_type'].upper()}")
        print(f"Prediction:   {row['prediction']}")
        print(f"Risk Score:   {row['risk_score']}")
        print(f"Timestamp:    {row['timestamp']}")
        
        while True:
            choice = input("Accept and append to dataset? (y = Yes / n = No / s = Skip): ").strip().lower()
            if choice in ['y', 'n', 's']:
                break
            print("Invalid choice.")
            
        if choice == 'y':
            # Map feedback to dataset label
            label = 1 if row['feedback_type'] == 'false_negative' else 0
            accepted_entries.append({
                'url': row['url_identifier'],
                'label': label,
                'source': 'user_feedback'
            })
            print("-> Accepted.")
        elif choice == 'n':
            print("-> Rejected.")
        elif choice == 's':
            print("-> Skipped.")

    if accepted_entries:
        # Load current dataset
        if os.path.exists(DATASET_FILE):
            dataset_df = pd.read_csv(DATASET_FILE)
        else:
            dataset_df = pd.DataFrame(columns=['url', 'label', 'source'])
            
        new_df = pd.DataFrame(accepted_entries)
        
        # Determine new version
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        versioned_filename = f'data/training_urls_v_{timestamp}.csv'
        
        # Combine and deduplicate
        combined_df = pd.concat([dataset_df, new_df], ignore_index=True)
        combined_df.drop_duplicates(subset=['url'], keep='last', inplace=True)
        
        # Save versioned dataset
        combined_df.to_csv(versioned_filename, index=False)
        # Update main dataset symlink/copy
        combined_df.to_csv(DATASET_FILE, index=False)
        
        print("-" * 50)
        print(f"Successfully appended {len(accepted_entries)} verified URLs to the dataset.")
        print(f"Dataset version saved as: {versioned_filename}")
        print("Note: The model is NOT automatically retrained. Run 'scripts/train_model.py' when ready.")
        
        # Clear actionable feedback we processed? 
        # (For MVP, we just leave it in the CSV or clear it manually)
    else:
        print("-" * 50)
        print("No feedback was accepted. Dataset remains unchanged.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process PhishGuard user feedback")
    parser.parse_args()
    review_feedback()
