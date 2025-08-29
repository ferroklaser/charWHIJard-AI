import os
import argparse
import pandas as pd
import re

def process_data(meta_path, review_path, output_dir="../data/processed"):
    meta_df = pd.read_json(meta_path, lines=True, compression="gzip")
    review_df = pd.read_json(review_path, lines=True, compression="gzip")

    review_df = review_df[["text", "gmap_id"]]

    # Drop reviews with no text
    review_df = review_df.dropna(subset=["text"])
    meta_df["category"] = meta_df["category"].apply(
        lambda x: ", ".join(x) if isinstance(x, list) else x
    )

    # Join with metadata 
    df = review_df.merge(
        meta_df[["gmap_id", "name", "category"]],
        on="gmap_id",
        how="left"
    )

    df = df[["name", "category", "text"]].drop_duplicates()

    os.makedirs(output_dir, exist_ok=True)

    base_name = os.path.basename(review_path)
    match = re.match(r"review-(.+)\.json\.gz", base_name)
    if not match:
        raise ValueError(f"Unexpected filename format: {base_name}")
    
    region = match.group(1).lower()
    output_filename = f"{region}.csv"

    output_path = os.path.join(output_dir, output_filename)
    df.to_csv(output_path, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("meta", help="Path to meta-<region>.json.gz")
    parser.add_argument("review", help="Path to review-<region>.json.gz")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))

    meta_path = os.path.join(script_dir, "../data/raw", args.meta)
    review_path = os.path.join(script_dir, "../data/raw", args.review)

    process_data(meta_path, review_path)