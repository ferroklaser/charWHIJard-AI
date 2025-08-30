import pandas as pd
from sklearn.metrics import precision_recall_fscore_support
import argparse

parser = argparse.ArgumentParser(description="Evaluate Qwen vs GPT-4.1-nano labels")
parser.add_argument("baseline_csv", help="Path to baseline CSV (GPT-4.1-nano labels)")
parser.add_argument("qwen_csv", help="Path to Qwen predictions CSV")
args = parser.parse_args()

LABELS = ["is_relevant", "is_advertisement", "is_rant_without_review"]

baseline = pd.read_csv(args.baseline_csv)
qwen = pd.read_csv(args.qwen_csv)

# Convert relevancy_score -> is_relevant (binary)
# Setting all relevancy scores > 0.5 as relevant and those below as irrelevant
baseline["is_relevant"] = (baseline["relevancy_score"] >= 0.5).astype(int)
qwen["is_relevant"] = (qwen["relevancy_score"] >= 0.5).astype(int)

# Standardise the boolean columns (is_rant_without_review and is_advertisement)
def normalize_bool(col):
    def convert(x):
        if isinstance(x, bool):
            return int(x)  # True -> 1, False -> 0
        x_str = str(x).strip().lower()
        return 1 if x_str == "true" else 0

    return col.apply(convert)

for col in ["is_advertisement", "is_rant_without_review"]:
    baseline[col] = normalize_bool(baseline[col])
    qwen[col] = normalize_bool(qwen[col])

def evaluate(y_true, y_pred):
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    return precision, recall, f1

rows = []
all_true = []
all_pred = []

for col in LABELS:
    y_true = baseline[col]
    y_pred = qwen[col]
    p, r, f1 = evaluate(y_true, y_pred)
    rows.append({
        "Label": col,
        "Precision": round(p, 3),
        "Recall": round(r, 3),
        "F1": round(f1, 3)
    })
    all_true.append(y_true)
    all_pred.append(y_pred)

y_true_micro = pd.concat(all_true)
y_pred_micro = pd.concat(all_pred)

p_micro, r_micro, f1_micro, _ = precision_recall_fscore_support(
    y_true_micro, y_pred_micro, average="micro", zero_division=0
)
p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
    y_true_micro, y_pred_micro, average="macro", zero_division=0
)

rows.append({"Label": "MICRO_AVG", "Precision": round(p_micro, 3), "Recall": round(r_micro, 3), "F1": round(f1_micro, 3)})
rows.append({"Label": "MACRO_AVG", "Precision": round(p_macro, 3), "Recall": round(r_macro, 3), "F1": round(f1_macro, 3)})

# Displaying results in table form
results_df = pd.DataFrame(rows)
print("\nQwen vs GPT-4.1-nano (Baseline) Evaluation")
print(results_df.to_string(index=False))