"""Analyse Classifier A results and exact feature overlap with training."""

import hashlib
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"
OUT = ROOT / "data" / "output"

plt.rcParams.update({"font.size": 11})


def save_figure(fig: plt.Figure, name: str) -> None:
    """Save a report figure and release its memory."""
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=200, bbox_inches="tight")
    plt.close(fig)


def read_predictions(split: str) -> pd.DataFrame:
    """Load predictions and verify their recorded correctness."""
    df = pd.read_csv(OUT / f"classifier_a_{split}_predictions.csv")
    if not np.array_equal(df["row_index"], np.arange(len(df))):
        raise ValueError(f"{split}: prediction rows are not in array order.")

    df["correct_top1"] = df["predicted_label"] == df["true_label"]
    labels = df[[f"top{k}_label" for k in range(1, 6)]].to_numpy()
    df["correct_top5"] = (
        labels == df["true_label"].to_numpy()[:, None]
    ).any(axis=1)
    return df


def feature_hash(row: np.ndarray) -> bytes:
    """Hash the exact stored feature values for overlap screening."""
    return hashlib.sha256(np.asarray(row).tobytes()).digest()


def main() -> None:
    """Generate descriptive figures and an overlap sensitivity analysis."""
    OUT.mkdir(parents=True, exist_ok=True)
    predictions = {
        split: read_predictions(split)
        for split in ("test1", "test2")
    }

    # 1. Learning curves: internal validation only.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for alpha in ("1e-05", "0.0001", "0.001"):
        history = pd.read_csv(
            OUT / f"classifier_a_dev_n100_alpha{alpha}_epochs5.csv"
        )
        axes[0].plot(
            history["epoch"],
            100 * history["val_top1"],
            marker="o",
            label=f"alpha={alpha}",
        )

    full = pd.read_csv(
        OUT / "classifier_a_full_alpha1e-05_epochs10.csv"
    )
    axes[1].plot(
        full["epoch"],
        100 * full["val_top1"],
        marker="o",
    )
    best = full.loc[full["val_top1"].idxmax()]
    axes[1].scatter(
        [best["epoch"]],
        [100 * best["val_top1"]],
        color="red",
        zorder=3,
        label="Selected checkpoint",
    )

    axes[0].set_title("100,000-example tuning runs")
    axes[1].set_title("Full training: alpha=1e-05")
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Internal validation top-1 (%)")
        ax.legend()
        ax.grid(alpha=0.2)
    save_figure(fig, "fig_classifier_a_learning.png")

    # 2. Accuracy comparison, recomputed from predictions.
    metrics = pd.DataFrame([
        {
            "test_set": split,
            "rows": len(df),
            "top1": df["correct_top1"].mean(),
            "top5": df["correct_top5"].mean(),
        }
        for split, df in predictions.items()
    ])
    metrics.to_csv(OUT / "classifier_a_analysis_metrics.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(2)
    for offset, metric, label in (
        (-0.18, "top1", "Top-1"),
        (0.18, "top5", "Top-5"),
    ):
        bars = ax.bar(x + offset, 100 * metrics[metric], 0.36, label=label)
        ax.bar_label(bars, fmt="%.2f%%", padding=3)
    ax.set_xticks(x, ["ImageNet", "ImageNetV2"])
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 108)
    ax.set_title("Classifier A: final test accuracy")
    ax.legend(loc="lower left")
    save_figure(fig, "fig_classifier_a_test_accuracy.png")

    # 3. Per-class comparison. V2 has only ten examples per class.
    per_class = []
    for split, df in predictions.items():
        table = df.groupby("true_label").agg(
            rows=("correct_top1", "size"),
            top1=("correct_top1", "mean"),
        )
        per_class.append(table.add_prefix(f"{split}_"))

    comparison = per_class[0].join(per_class[1], how="outer")
    if comparison.isna().any().any():
        raise ValueError("Class coverage differs between test sets.")
    comparison["gap_pp"] = 100 * (
        comparison["test1_top1"] - comparison["test2_top1"]
    )
    comparison.to_csv(OUT / "classifier_a_per_class.csv")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].scatter(
        100 * comparison["test1_top1"],
        100 * comparison["test2_top1"],
        alpha=0.25,
        s=15,
    )
    axes[0].plot([0, 100], [0, 100], "--", color="grey")
    axes[0].set_xlabel("ImageNet class accuracy (%)")
    axes[0].set_ylabel("ImageNetV2 class accuracy (%)")
    axes[0].set_title("Per-class accuracy")

    axes[1].hist(comparison["gap_pp"], bins=25, edgecolor="white")
    axes[1].axvline(0, linestyle="--", color="grey")
    axes[1].set_xlabel("ImageNet minus ImageNetV2 (percentage points)")
    axes[1].set_ylabel("Number of classes")
    axes[1].set_title("Per-class gaps: descriptive, small V2 samples")
    save_figure(fig, "fig_classifier_a_class_gaps.png")

    # 4. Exact feature overlap with the rows actually used for training.
    print("Checking test1 overlap with actual training rows...", flush=True)
    X_test = np.load(PROC / "test1_X.npy", mmap_mode="r")
    X_train = np.load(PROC / "train_full_X.npy", mmap_mode="r")
    train_idx = np.load(PROC / "split_train_idx.npy")

    df = predictions["test1"]
    y_test = np.load(PROC / "test1_y.npy")
    if len(X_test) != len(df):
        raise ValueError("Test1 feature and prediction lengths differ.")
    if not np.array_equal(y_test, df["true_label"].to_numpy()):
        raise ValueError("Test1 labels and prediction labels differ.")

    candidates = {}
    for i, row in enumerate(X_test):
        candidates.setdefault(feature_hash(row), []).append(i)

    matched = np.zeros(len(X_test), dtype=bool)
    first_train_match = np.full(len(X_test), -1, dtype=np.int64)

    for start in range(0, len(train_idx), 10000):
        indices = train_idx[start:start + 10000]
        batch = X_train[indices]

        for source_index, row in zip(indices, batch):
            for test_index in candidates.get(feature_hash(row), ()):
                # Verify exact equality rather than relying on hashes alone.
                if np.array_equal(row, X_test[test_index]):
                    matched[test_index] = True
                    if first_train_match[test_index] == -1:
                        first_train_match[test_index] = source_index

        if start % 100000 == 0:
            print(
                f"Checked {min(start + 10000, len(train_idx)):,}"
                f"/{len(train_idx):,} training rows",
                flush=True,
            )

    overlap = df[
        ["row_index", "path", "true_label", "predicted_label", "correct_top1"]
    ].copy()
    overlap["matches_training_features"] = matched
    overlap["first_matching_train_row"] = first_train_match
    overlap.to_csv(OUT / "classifier_a_test1_overlap.csv", index=False)

    groups = [
        ("test1_all", df),
        ("test1_matching_training", df.loc[matched]),
        ("test1_no_training_match", df.loc[~matched]),
        ("test2_all", predictions["test2"]),
    ]
    sensitivity = pd.DataFrame([
        {
            "group": name,
            "rows": len(group),
            "top1": group["correct_top1"].mean(),
            "top5": group["correct_top5"].mean(),
        }
        for name, group in groups
    ])
    sensitivity.to_csv(
        OUT / "classifier_a_overlap_sensitivity.csv", index=False
    )

    print("\nOverall metrics:")
    print(metrics.to_string(index=False))
    print("\nOverlap sensitivity:")
    print(sensitivity.to_string(index=False))
    print(f"\nFigures and analysis saved in: {OUT}")


if __name__ == "__main__":
    main()