# 03_mlp_classifier.py - main task: a small MLP on EVA-02 features (1,024-d), tuned on our validation set

import itertools
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 11})

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"
OUT = ROOT / "data" / "output"
OUT.mkdir(parents=True, exist_ok=True)

SEED = 7
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
BATCH = 1024
TUNE_EPOCHS = 10          # per setting, on the 100,000-row train_small set
FINAL_EPOCHS = 20         # for the final model, on all training rows
GRID = {"hidden": [512, 1024], "dropout": [0.0, 0.5], "weight_decay": [0.0, 0.1]}


# 1. Load
# train_small is for tuning; train_full minus our validation rows is for the final model
# the test sets are loaded only at the end and are never used to fit or choose anything
def load(name):
    return np.load(PROC / f"{name}_X.npy"), np.load(PROC / f"{name}_y.npy").astype(np.int64)


X_small, y_small = load("train_small")
X_val, y_val = load("val")
X_full, y_full = load("train_full")
idx = np.load(PROC / "split_train_idx.npy")
X_full, y_full = X_full[idx], y_full[idx]
print("train_small:", X_small.shape, "train_full:", X_full.shape, "val:", X_val.shape, "device:", DEVICE)

# standardise every feature with the mean and std of the training data
mean, std = X_full.mean(axis=0), X_full.std(axis=0) + 1e-6


def to_tensor(X):
    return torch.from_numpy((X - mean) / std)


# 2. Model and training
# one hidden layer, trained with mini-batches (Lab 8)
def make_mlp(hidden, dropout):
    return nn.Sequential(nn.Linear(1024, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, 1000))


def predict(model, X):
    # the 5 highest-scoring classes of every row
    model.eval()
    with torch.no_grad():
        return torch.cat([model(X[i:i + 10_000].to(DEVICE)).topk(5, dim=1).indices.cpu()
                          for i in range(0, len(X), 10_000)]).numpy()


def accuracy(top5, y):
    return (top5[:, 0] == y).mean(), (top5 == y[:, None]).any(axis=1).mean()


def train(X, y, hidden, dropout, weight_decay, epochs):
    # returns the model of the epoch with the best validation top-1, and the score of every epoch
    torch.manual_seed(SEED)
    model = make_mlp(hidden, dropout).to(DEVICE)
    optimiser = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()
    y = torch.from_numpy(y)
    history, best, best_state = [], -1, None
    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(len(X))
        total = 0
        for i in range(0, len(X), BATCH):
            batch = order[i:i + BATCH]
            loss = loss_fn(model(X[batch].to(DEVICE)), y[batch].to(DEVICE))
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            total += loss.item() * len(batch)
        top1, top5 = accuracy(predict(model, Xs_val), y_val)
        history.append({"epoch": epoch, "train_loss": total / len(X), "val_top1": top1, "val_top5": top5})
        if top1 > best:
            best, best_state = top1, {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model, pd.DataFrame(history)


# 3. Tune
# every combination in GRID is trained on train_small and scored on our validation set
Xs_small, Xs_val = to_tensor(X_small), to_tensor(X_val)
tuning = []
for hidden, dropout, weight_decay in itertools.product(GRID["hidden"], GRID["dropout"], GRID["weight_decay"]):
    t0 = time.time()
    _, history = train(Xs_small, y_small, hidden, dropout, weight_decay, TUNE_EPOCHS)
    best = history.loc[history.val_top1.idxmax()]
    tuning.append({"hidden": hidden, "dropout": dropout, "weight_decay": weight_decay, "best_epoch": int(best.epoch),
                   "val_top1": best.val_top1, "val_top5": best.val_top5, "seconds": time.time() - t0})
    print(tuning[-1])
tuning = pd.DataFrame(tuning)
tuning.to_csv(OUT / "mlp_tuning.csv", index=False)
chosen = tuning.loc[tuning.val_top1.idxmax()]
print("chosen:", chosen[["hidden", "dropout", "weight_decay"]].to_dict())


# 4. Final model
# the chosen setting, trained on all training rows
t0 = time.time()
model, history = train(to_tensor(X_full), y_full, int(chosen.hidden), chosen.dropout, chosen.weight_decay, FINAL_EPOCHS)
train_seconds = time.time() - t0
history.to_csv(OUT / "mlp_final_history.csv", index=False)
print(history.round(4).to_string(index=False))


# 5. Evaluate on both test sets
# this is done once, after every choice above is made
results, predictions = [], []
for name, label in [("test1", "test set 1 (ImageNet val)"), ("test2", "test set 2 (ImageNetV2)")]:
    X_test, y_test = load(name)
    t0 = time.time()
    top5 = predict(model, to_tensor(X_test))
    top1_acc, top5_acc = accuracy(top5, y_test)
    results.append({"model": "MLP", "test_set": label, "rows": len(y_test), "top1": top1_acc, "top5": top5_acc,
                    "train_seconds": train_seconds, "predict_seconds": time.time() - t0})
    predictions.append(pd.DataFrame({"test_set": name, "row": np.arange(len(y_test)), "label": y_test, "pred": top5[:, 0]}))
results = pd.DataFrame(results)
results.to_csv(OUT / "mlp_results.csv", index=False)
pd.concat(predictions).to_csv(OUT / "mlp_predictions.csv", index=False)
print(results.round(4).to_string(index=False))
print("top-1 gap (test set 1 - test set 2): %.1f points" % (100 * (results.top1[0] - results.top1[1])))

fig, ax = plt.subplots(figsize=(6.2, 3.4))
ax.plot(history.epoch, history.val_top1, marker="o", color="#2a78d6", label="validation top-1")
ax.plot(history.epoch, history.val_top5, marker="o", color="#eb6834", label="validation top-5")
ax.set_xlabel("epoch")
ax.set_ylabel("accuracy")
ax.set_title("MLP on EVA-02 features: our validation set")
ax.legend(frameon=False)
ax.spines[["top", "right"]].set_visible(False)
plt.savefig(OUT / "fig_mlp_validation.png", dpi=150, bbox_inches="tight")
plt.show()
