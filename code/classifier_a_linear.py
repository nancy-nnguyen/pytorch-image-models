"""Classifier A: L2-regularised linear classification of EVA-02 features."""

import argparse
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"
OUT = ROOT / "data" / "output"

SEED = 7
BATCH_SIZE = 1024


def load_array(name: str) -> np.ndarray:
    """Load an array without reading the entire file into memory."""
    return np.load(PROC / f"{name}.npy", mmap_mode="r")


def evaluate(
        model: SGDClassifier,
        scaler: StandardScaler,
        X: np.ndarray,
        y: np.ndarray,
) -> tuple[float, float]:
    """Calculate top-1 and top-5 accuracy in bounded batches."""
    correct1 = 0
    correct5 = 0

    for start in range(0, len(y), BATCH_SIZE):
        stop = start + BATCH_SIZE
        batch = scaler.transform(X[start:stop])
        scores = model.decision_function(batch)
        labels = y[start:stop]

        predictions = model.classes_[np.argmax(scores, axis=1)]
        top5_columns = np.argpartition(scores, -5, axis=1)[:, -5:]
        top5_labels = model.classes_[top5_columns]

        correct1 += np.count_nonzero(predictions == labels)
        correct5 += np.count_nonzero(
            np.any(top5_labels == labels[:, None], axis=1)
        )

    return correct1 / len(y), correct5 / len(y)


def main() -> None:
    """Train on a stratified subset and evaluate internal validation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-class", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--alpha", type=float, default=0.0001)
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--overwrite", action="store_true",
                        help="Replace this run's existing history and checkpoint.")
    args = parser.parse_args()

    if not 1 <= args.per_class <= 100:
        parser.error("--per-class must be between 1 and 100")
    if args.epochs < 1 or args.alpha <= 0:
        parser.error("--epochs and --alpha must be positive")
    if args.patience < 1:
        parser.error("--patience must be positive")

    X_val = load_array("val_X")
    y_val = load_array("val_y")
    rng = np.random.default_rng(SEED)

    if args.full:
        X = load_array("train_full_X")
        y = load_array("train_full_y")
        selected = np.load(PROC / "split_train_idx.npy")
        validation_indices = np.load(PROC / "split_val_idx.npy")

        if selected.ndim != 1 or len(selected) == 0:
            raise ValueError("Training indices must be a non-empty vector.")
        if selected.min() < 0 or selected.max() >= len(y):
            raise ValueError("Training indices are outside the array.")
        if len(np.unique(selected)) != len(selected):
            raise ValueError("Training indices contain duplicates.")
        if np.intersect1d(selected, validation_indices).size:
            raise ValueError("Training and validation indices overlap.")
        if not np.array_equal(y[validation_indices], y_val):
            raise ValueError("Validation labels do not match the shared split.")

        mode = "full"
    else:
        X = load_array("train_small_X")
        y = load_array("train_small_y")
        selected = np.concatenate([
            rng.choice(
                np.flatnonzero(y == label),
                size=args.per_class,
                replace=False,
            )
            for label in np.unique(y)
        ])
        mode = f"n{args.per_class}"

    if X.shape != (len(y), 1024) or X_val.shape != (len(y_val), 1024):
        raise ValueError("Unexpected feature or label dimensions.")

    classes = np.unique(y[selected])
    if not np.array_equal(classes, np.arange(1000)):
        raise ValueError("Expected training labels 0 through 999.")

    OUT.mkdir(parents=True, exist_ok=True)
    model_dir = ROOT / "data" / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    prefix = "classifier_a_full" if args.full else f"classifier_a_dev_{mode}"
    run_name = f"{prefix}_alpha{args.alpha:g}_epochs{args.epochs}"
    history_path = OUT / f"{run_name}.csv"
    checkpoint_path = model_dir / f"{run_name}_best.joblib"

    if (history_path.exists() or checkpoint_path.exists()) and not args.overwrite:
        raise FileExistsError(
            f"Outputs already exist for {run_name}. "
            "Use --overwrite to regenerate this run, or back up its outputs first."
        )

    print(
        f"Training rows: {len(selected):,}; "
        f"validation rows: {len(y_val):,}; alpha={args.alpha}",
        flush=True,
    )

    scaler = StandardScaler()
    model = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=args.alpha,
        random_state=SEED,
        n_jobs=4,
    )

    history = []
    best_top1 = -1.0
    best_epoch = 0
    stale_epochs = 0
    total_start = time.perf_counter()

    with threadpool_limits(limits=4):
        print("Fitting scaler on training-only rows...", flush=True)
        for start in range(0, len(selected), BATCH_SIZE):
            rows = selected[start:start + BATCH_SIZE]
            batch = np.asarray(X[rows])
            if not np.isfinite(batch).all():
                raise ValueError("Training features contain non-finite values.")
            scaler.partial_fit(batch)

        print("Scaler fitted. Starting training...", flush=True)

        for epoch in range(1, args.epochs + 1):
            epoch_start = time.perf_counter()
            order = rng.permutation(selected)

            for batch_number, start in enumerate(
                range(0, len(order), BATCH_SIZE), start=1
            ):
                rows = order[start:start + BATCH_SIZE]
                batch = scaler.transform(X[rows])
                model.partial_fit(batch, y[rows], classes=classes)

                if batch_number % 200 == 0:
                    print(
                        f"epoch={epoch} "
                        f"trained={min(start + BATCH_SIZE, len(order)):,}"
                        f"/{len(order):,}",
                        flush=True,
                    )

            top1, top5 = evaluate(model, scaler, X_val, y_val)
            improved = top1 > best_top1

            if improved:
                best_top1 = top1
                best_epoch = epoch
                stale_epochs = 0
                joblib.dump(
                    {
                        "model": model,
                        "scaler": scaler,
                        "seed": SEED,
                        "alpha": args.alpha,
                        "mode": mode,
                        "training_rows": len(selected),
                        "best_epoch": epoch,
                        "val_top1": top1,
                        "val_top5": top5,
                        "learning_rate": model.learning_rate,
                    },
                    checkpoint_path,
                )
            else:
                stale_epochs += 1

            seconds = time.perf_counter() - epoch_start
            history.append({
                "seed": SEED,
                "mode": mode,
                "training_rows": len(selected),
                "alpha": args.alpha,
                "epoch": epoch,
                "val_top1": top1,
                "val_top5": top5,
                "seconds": seconds,
                "best_so_far": improved,
            })
            pd.DataFrame(history).to_csv(history_path, index=False)

            print(
                f"epoch={epoch} val_top1={top1:.5f} "
                f"val_top5={top5:.5f} seconds={seconds:.1f} "
                f"best_epoch={best_epoch}",
                flush=True,
            )

            if stale_epochs >= args.patience:
                print("Early stopping: validation top-1 stopped improving.")
                break

    print(f"Best validation top-1: {best_top1:.5f} at epoch {best_epoch}")
    print(f"Total minutes: {(time.perf_counter() - total_start) / 60:.1f}")
    print(f"History: {history_path}")
    print(f"Best model and scaler: {checkpoint_path}")



if __name__ == "__main__":
    main()