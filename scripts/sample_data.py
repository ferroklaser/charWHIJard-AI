import os
import argparse
import pandas as pd

def sample_reviews(input_csv, n, output_dir="data/samples"):
    # Read processed CSV
    df = pd.read_csv(input_csv)

    if "text" not in df.columns:
        raise ValueError("CSV must have a column named 'text'")

    df_sample = df.sample(n=n, random_state=42)

    os.makedirs(output_dir, exist_ok=True)

    base_name = os.path.basename(input_csv)
    if base_name.endswith(".csv"):
        base_name = base_name[:-4]
    output_csv = os.path.join(output_dir, f"{base_name}_sample_{n}.csv")

    df_sample.to_csv(output_csv, index=False)
    print(f"Sampled {n} reviews saved to {output_csv}")
    return df_sample

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sampled reviews from processed CSV")
    parser.add_argument("input_file", help="Path to CSV file (any location)")
    parser.add_argument("--n", type=int, default=100, help="Number of reviews to sample (default=200)")
    args = parser.parse_args()

    input_path = os.path.join("data", "processed", args.input_file)
    sample_reviews(args.input_file, n=args.n)