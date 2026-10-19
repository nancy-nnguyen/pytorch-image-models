"""Evaluate the frozen Classifier A checkpoint on both assignment test sets."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"
OUT = ROOT / "data" / "output"
CHECKPOINT = (
    ROOT / "data" / "models"
    / "classifier_a_full_alpha1e-05_epochs10_best.joblib"
)
BATCH_SIZE = 1024


def main() -> None:
    """Export test metrics and predictions without fitting anything."""
    # Load only the checkpoint produced by our own training script.
    saved = joblib.load(CHECKPOINT)
    model = saved["model"]
    scaler = saved["scaler"]
    OUT.mkdir(parents=True, exist_ok=True)

    print(
        f"Loaded epoch {saved['best_epoch']}; "
        f"validation top-1={saved['val_top1']:.5f}",
        flush=True,
    )

    summary = []

    with threadpool_limits(limits=4):
        for split in ("test1", "test2"):
            X = np.load(PROC / f"{split}_X.npy", mmap_mode="r")
            y = np.load(PROC / f"{split}_y.npy")
            meta = pd.read_csv(PROC / f"{split}_meta.csv")

            if X.shape != (len(y), model.n_features_in_):
                raise ValueError(f"{split}: unexpected feature dimensions.")
            if len(meta) != len(y):
                raise ValueError(f"{split}: metadata length mismatch.")
            if not np.array_equal(meta["label"].to_numpy(), y):
                raise ValueError(f"{split}: metadata label mismatch.")
            if not np.isin(y, model.classes_).all():
                raise ValueError(f"{split}: unknown labels.")

            batches = []

            for start in range(0, len(y), BATCH_SIZE):
                stop = min(start + BATCH_SIZE, len(y))
                features = np.asarray(X[start:stop])

                if not np.isfinite(features).all():
                    raise ValueError(f"{split}: non-finite features.")

                scores = model.decision_function(
                    scaler.transform(features)
                )
                if not np.isfinite(scores).all():
                    raise ValueError(f"{split}: non-finite scores.")

                # Sort the five highest-scoring classes in descending order.
                columns = np.argpartition(scores, -5, axis=1)[:, -5:]
                values = np.take_along_axis(scores, columns, axis=1)
                order = np.argsort(-values, axis=1)
                columns = np.take_along_axis(columns, order, axis=1)
                values = np.take_along_axis(values, order, axis=1)
                labels = model.classes_[columns]

                result = pd.DataFrame({
                    "row_index": np.arange(start, stop),
                    "path": meta["path"].iloc[start:stop].to_numpy(),
                    "true_label": y[start:stop],
                    "predicted_label": labels[:, 0],
                    "correct_top1": labels[:, 0] == y[start:stop],
                    "correct_top5": (
                        labels == y[start:stop, None]
                    ).any(axis=1),
                    "decision_margin": values[:, 0] - values[:, 1],
                })

                for rank in range(5):
                    result[f"top{rank + 1}_label"] = labels[:, rank]
                    result[f"top{rank + 1}_score"] = values[:, rank]

                batches.append(result)

            predictions = pd.concat(batches, ignore_index=True)
            top1 = float(predictions["correct_top1"].mean())
            top5 = float(predictions["correct_top5"].mean())

            predictions.to_csv(
                OUT / f"classifier_a_{split}_predictions.csv",
                index=False,
            )
            summary.append({
                "test_set": split,
                "rows": len(y),
                "top1": top1,
                "top5": top5,
                "alpha": saved["alpha"],
                "best_epoch": saved["best_epoch"],
                "seed": saved["seed"],
                "training_rows": saved["training_rows"],
            })
            print(
                f"{split}: rows={len(y):,} "
                f"top1={top1:.5f} top5={top5:.5f}",
                flush=True,
            )

    results = pd.DataFrame(summary)
    results.to_csv(OUT / "classifier_a_test_results.csv", index=False)

    gap = 100 * (summary[0]["top1"] - summary[1]["top1"])
    print(f"Top-1 gap (test1 - test2): {gap:.3f} percentage points")
    print(f"Results saved in: {OUT}")


if __name__ == "__main__":
    main()