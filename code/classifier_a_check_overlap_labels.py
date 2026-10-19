"""Check label agreement for the first exact training match of each test1 row.

Run after classifier_a_analyse.py. This is descriptive analysis only; it does
not change labels, training data, or model selection.
"""

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

# The analysis output records the first verified feature match in training.
df = pd.read_csv(ROOT / "data/output/classifier_a_test1_overlap.csv")
matched = df.loc[df["first_matching_train_row"] >= 0].copy()

y_train = np.load(ROOT / "data/processed/train_full_y.npy")

# Compare test labels with the recorded training-row labels.
indices = matched["first_matching_train_row"].to_numpy(dtype=np.int64)

matched["matching_train_label"] = y_train[indices]
matched["labels_agree"] = (
    matched["true_label"] == matched["matching_train_label"]
)
matched["prediction_matches_train_label"] = (
    matched["predicted_label"] == matched["matching_train_label"]
)

matched.to_csv(
    ROOT / "data/output/classifier_a_overlap_label_check.csv",
    index=False,
)

print("Matching test examples:", len(matched))
print("Same label:", int(matched["labels_agree"].sum()))
print("Different label:", int((~matched["labels_agree"]).sum()))
print(
    "Prediction matches first matching training label:",
    int(matched["prediction_matches_train_label"].sum()),
)
