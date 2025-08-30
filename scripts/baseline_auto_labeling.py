import pandas as pd
import os
import argparse
import re
import json
import time
from openai import OpenAI


# EXAMPLE in terminal: python scripts/baseline_auto_labeling.py data/samples/south_dakota_sample.csv

# Initialize OpenAI client using API key (we will be using gpt-4.1-nano for the baseline)
client = OpenAI(api_key="xxx")  # replace with working key

def gpt_label_review(text: str):
    """
    Use GPT-4.1-nano to label a review with:
    - relevancy_score (0.0-1.0)
    - is_advertisement (bool)
    - is_rant_without_review (bool)
    """

    # Filter out obvious spam and adverts with simple logic checks
    if len(text.split()) < 3:  # very short so most likely irrelevant
        return {"relevancy_score": 0.0, "is_advertisement": False, "is_rant_without_review": True}
    if "http" in text or "www" in text: # likely an advert
        return {"relevancy_score": 0.0, "is_advertisement": True, "is_rant_without_review": False}

    safe_text = text.replace('\\', '\\\\').replace('"', '\\"').replace('\n', ' ')
    prompt = f"""
    You are labeling restaurant reviews with three fields:
    1. relevancy_score: from 0.0 (not relevant at all) to 1.0 (highly relevant).
    2. is_advertisement: True if the text looks like spam, promotion, or contains website links (ads). Otherwise False.
    3. is_rant_without_review: True if it's just angry ranting, insults, or nonsense with no actual review of food/service. Otherwise False.

    Respond in JSON only, with keys: relevancy_score, is_advertisement, is_rant_without_review.

    Review: "{safe_text}"
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        content = response.choices[0].message.content.strip()
        # Extract JSON if GPT added extra text
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        else:
            return {"relevancy_score": 0.0, "is_advertisement": False, "is_rant_without_review": False}
    except Exception as e:
        print(f"GPT failed: {e}")
        return {"relevancy_score": 0.0, "is_advertisement": False, "is_rant_without_review": False}

def auto_label_reviews(input_csv, sample_n=200, sleep_time=1):
    df = pd.read_csv(input_csv)

    if 'text' not in df.columns:
        raise ValueError("CSV must have a column named 'text'")

    if sample_n and len(df) > sample_n:
        df = df.sample(n=sample_n, random_state=42)

    print(f"Labeling {len(df)} reviews...")

    all_labels = []
    for idx, text in enumerate(df['text'], 1):
        label = gpt_label_review(text)
        all_labels.append(label)
        print(f"[{idx}/{len(df)}] Labeled review: {label}")

        # debugging for in case there are some errors with the output rate
        # time.sleep(sleep_time)

    labels_df = pd.json_normalize(all_labels)

    df_out = labels_df

    base_name = os.path.basename(input_csv).replace(".csv", "")
    output_dir = os.path.join(os.path.dirname(input_csv), "baseline_result")
    os.makedirs(output_dir, exist_ok=True)
    output_csv = os.path.join(output_dir, f"{base_name}_gpt_baseline.csv")
    df_out.to_csv(output_csv, index=False)
    print(f"Saved baseline CSV to: {output_csv}")
    return df_out

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hybrid GPT+rules baseline labeling")
    parser.add_argument("input_file", help="CSV file with 'text' column")
    parser.add_argument("--n", type=int, default=200, help="Number of reviews to sample")
    args = parser.parse_args()

    auto_label_reviews(args.input_file, sample_n=args.n)