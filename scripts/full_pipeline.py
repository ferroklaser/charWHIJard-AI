import os
import pandas as pd

from download_raw_data import download_raw_data
from process_data import preprocess_all
from baseline_auto_labeling import auto_label_reviews
from classify_qwen import classify_review_qwen
from evaluate_tests import evaluate_labels

RAW_DIR = "../data/raw"
PROCESSED_DIR = "../data/processed"
LABELED_DIR = "../data/labeled"

os.makedirs(LABELED_DIR, exist_ok=True)

def run_pipeline():
    # Download raw datasets
    download_raw_data(RAW_DIR)

    # Preprocess raw data
    processed_files = preprocess_all(RAW_DIR, PROCESSED_DIR)
    
    for input_csv in processed_files:
        # GPT-4.1-nano baseline labelling
        baseline_df, baseline_csv = auto_label_reviews(input_csv, sample_n=200)

        df = pd.read_csv(input_csv)
        scores, ads, rants = [], [], []
        for _, row in df.iterrows():
            score, is_ad, is_rant = classify_review_qwen(row)
            scores.append(score)
            ads.append(is_ad)
            rants.append(is_rant)
        df["relevancy_score"] = scores
        df["is_advertisement"] = ads
        df["is_rant_without_review"] = rants

        qwen_csv = os.path.join(
            LABELED_DIR,
            f"{os.path.basename(input.csv).replace('.csv','')}_qwen_labels.csv"
        )
        df.to_csv(qwen_csv, index=False)
        results_df = evaluate_labels(baseline_csv, qwen_csv)
    print("Pipeline finished for all regions!")

if __name__ == "__maine__":
    run_pipeline()
