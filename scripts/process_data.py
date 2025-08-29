import os
import pandas as pd
import re
from collections import defaultdict

raw_dir = "../data/raw"
output_dir="../data/processed"

def process_data(meta_path, review_path, output_dir="../data/processed"):
    meta_df = pd.read_json(meta_path, lines=True, compression="gzip")
    review_df = pd.read_json(review_path, lines=True, compression="gzip")

    review_df = review_df[["text", "gmap_id"]]
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
    print(f"Processed {region}, saved to {output_path}")

regions = defaultdict(dict)

for fname in os.listdir(raw_dir):
    if fname.startswith("review-") and fname.endswith(".json.gz"):
        region = fname[len("review-"):-len(".json.gz")]
        regions[region]["review"] = os.path.join(raw_dir, fname)
    elif fname.startswith("meta-") and fname.endswith(".json.gz"):
        region = fname[len("meta-"):-len(".json.gz")]
        regions[region]["meta"] = os.path.join(raw_dir, fname)


for region, files in regions.items():
    if "meta" in files and "review" in files:
        output_filename = f"{region}.csv"
        output_path = os.path.join(output_dir, output_filename)

        if os.path.exists(output_path):
            print(f"Skipping {region} (already exists at {output_path})")
            continue

        process_data(files["meta"], files["review"], output_dir)
    else:
        print(f"[!] Skipping {region} (missing meta or review file)")