import pandas as pd

# Import your classify function from the previous code
# from classify_qwen import classify_review_qwen  # if saved in a separate file
from classify_qwen import classify_review_qwen


csv_file = "../data/labeled/south_dakota_labeled.csv"  # your CSV file path
df = pd.read_csv(csv_file)

N = 10
subset_df = df.head(N).copy()
# Create a new column for the predicted labels
subset_df['predicted_label'] = subset_df['text'].apply(classify_review_qwen)

# Save to a new CSV
subset_df.to_csv("../data/labeled/my_reviews_labeled.csv", index=False)

print("Finished classifying! Saved to data/labeled/my_reviews_labeled.csv")
