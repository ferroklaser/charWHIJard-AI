import pandas as pd
from transformers import pipeline
import argparse
import os

# this is used as our baseline to auto-label some of the reviews for model training
# creates a csv file that has only the text and label columns

# Simple classification of labels for now (baseline labelling function) -> edit this to include other labels
def map_to_policy_label(text, pred):
    label = pred[0]['label']
    if "www" in text or "http" in text:
        return "ad-like"
    elif label == "NEGATIVE":
        return "Negative"
    else:
        return "Positive"

# Main function for labeling
def auto_label_reviews(input_csv):
    df = pd.read_csv(input_csv)

    # 100 reviews first
    df = df.sample(n=100, random_state=42)
    
    if 'text' not in df.columns:
        raise ValueError("CSV must have a column named 'text'")
    
    classifier = pipeline(
        "text-classification",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        tokenizer="distilbert-base-uncased-finetuned-sst-2-english",
        truncation=True,   # cut long reviews
        max_length=512     # BERT’s limit
    )

    # Applying the baseline labelling function
    print(f"Auto-labeling {len(df)} reviews...")
    df['label'] = df['text'].apply(lambda t: map_to_policy_label(t, classifier(t)))
    
    df_hf = df[['text', 'label']]

    # Determine output file
    base_name = os.path.basename(input_csv)
    if "_cleaned" in base_name:
        output_name = base_name.replace("_cleaned", "_labeled")
    else:
        output_name = base_name + "_labeled" # fallback

    project_root = os.path.dirname(os.path.dirname(input_csv))
    output_dir = os.path.join(project_root, "labeled")
    os.makedirs(output_dir, exist_ok=True)

    output_csv = os.path.join(output_dir, output_name)
    
    df_hf.to_csv(output_csv, index=False)
    print(f"Auto-labeled CSV saved to {output_csv}")
    return df_hf

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Auto-label reviews for Hugging Face training")
    parser.add_argument("input_file", help="Name of CSV file in data/processed (e.g., south_dakota_cleaned.csv)")
    args = parser.parse_args()

    cleaned_folder = os.path.join("data", "cleaned")
    input_csv = os.path.join(cleaned_folder, args.input_file)
    
    auto_label_reviews(input_csv)