# 05_generalisation_analysis.py - why accuracy drops on test set 2 (ImageNetV2) vs test set 1 (ImageNet val)

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

plt.rcParams.update({"font.size": 11})

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"
OUT = ROOT / "data" / "output"
INFO = ROOT / "timm" / "data" / "_info"    # ImageNet class names, shipped with timm
OUT.mkdir(parents=True, exist_ok=True)

parser = argparse.ArgumentParser(description="A3 part (b): why accuracy drops on ImageNetV2")
parser.add_argument("--v2-images", type=Path, default=ROOT / "data" / "raw",
                    help="folder that holds imagenetv2-matched-frequency-format-val (the image figure is skipped "
                         "if it is not there)")
args = parser.parse_args()

SEED = 7
N_CLASSES = 1000
NULL_SPLITS = 20
TYPICALITY_BINS = 10
ETA_REPEATS = 5
PCA_DIMS, N_CLUSTERS = 128, 50
BLUE, ORANGE, GREY = "#2a78d6", "#eb6834", "#888888"

# A has the best validation top-1, so it is the one used from section 5 on
NAMES = {"A": "Regularised linear", "B": "Small MLP", "C": "Nearest centroid"}
PRIMARY = "A"


# 1. Labels, features and class names
y = {t: np.load(PROC / f"{t}_y.npy").astype(int) for t in ("test1", "test2")}
X = {t: np.load(PROC / f"{t}_X.npy") for t in ("test1", "test2")}
n = {t: len(y[t]) for t in y}
print("test1 rows:", n["test1"], " test2 rows:", n["test2"])

wnids = (INFO / "imagenet_synsets.txt").read_text().split()
lemmas = dict(line.split("\t", 1) for line in (INFO / "imagenet_synset_to_lemma.txt").read_text().splitlines())
class_name = np.array([lemmas[w].split(",")[0] for w in wnids])
meta1 = pd.read_csv(PROC / "test1_meta.csv")
if not (meta1.path.str.split("/").str[1].to_numpy() == np.array(wnids)[y["test1"]]).all():
    raise ValueError("Class names do not line up with the labels")


# 2. Load the predictions of each classifier
# each script saves its own format; all become an (n, k) array of classes, best first
def read_a(test):
    df = pd.read_csv(OUT / f"classifier_a_{test}_predictions.csv")
    return df.true_label.to_numpy(), df[[f"top{k}_label" for k in range(1, 6)]].to_numpy()


def read_b(test):
    df = pd.read_csv(OUT / "mlp_predictions.csv")
    df = df[df.test_set == test].sort_values("row")
    top = [c for c in ("pred1", "pred2", "pred3", "pred4", "pred5") if c in df] or ["pred"]
    return df.label.to_numpy(), df[top].to_numpy()


def read_c(test):
    df = pd.read_csv(OUT / f"ncm_predictions_{test}.csv").sort_values("row")
    return df.label.to_numpy(), df[[f"pred{k}" for k in range(1, 6)]].to_numpy()


preds = {}
for clf, reader in (("A", read_a), ("B", read_b), ("C", read_c)):
    try:
        loaded = {t: reader(t) for t in ("test1", "test2")}
    except FileNotFoundError as e:
        print(f"Skipping classifier {clf}: {e.filename} not found (run its script first)")
        continue
    for t, (labels, _) in loaded.items():
        if not np.array_equal(labels, y[t]):
            raise ValueError(f"Classifier {clf}, {t}: labels do not match {t}_y.npy (wrong, truncated or reordered file)")
    preds[clf] = {t: top for t, (_, top) in loaded.items()}
if PRIMARY not in preds:
    raise SystemExit(f"Classifier {PRIMARY} predictions are needed; run code/classifier_a_evaluate.py first")
classifiers = list(preds)
print("Loaded predictions for", [NAMES[c] for c in classifiers])


# 3. Overall accuracy and the gap
def gap_row(clf, label, correct1, correct2, top5_1=np.nan, top5_2=np.nan):
    p1, p2 = correct1.mean(), correct2.mean()
    se = np.sqrt(p1 * (1 - p1) / len(correct1) + p2 * (1 - p2) / len(correct2))
    return {"classifier": NAMES[clf], "test1_rows": len(correct1), "test1_top1": p1, "test2_top1": p2,
            "test1_top5": top5_1, "test2_top5": top5_2,
            "gap": p1 - p2, "gap_ci_low": p1 - p2 - 1.96 * se, "gap_ci_high": p1 - p2 + 1.96 * se, "subset": label}


correct = {c: {t: preds[c][t][:, 0] == y[t] for t in y} for c in classifiers}
rows = []
for c in classifiers:
    # NaN when only the top-1 was saved (the MLP)
    top5 = {t: (preds[c][t] == y[t][:, None]).any(axis=1).mean() if preds[c][t].shape[1] == 5 else np.nan
            for t in y}
    rows.append(gap_row(c, "all test1", correct[c]["test1"], correct[c]["test2"], top5["test1"], top5["test2"]))


# 4. Test set 1 images that are also in the training data
# identical features = identical image; each one carries a different label in train
def row_hash(r):
    return hashlib.md5(r.tobytes()).digest()


test1_hash = {}
for i, r in enumerate(X["test1"]):
    test1_hash.setdefault(row_hash(r), []).append(i)
X_train = np.load(PROC / "train_full_X.npy", mmap_mode="r")
duplicate = np.zeros(n["test1"], dtype=bool)
for start in range(0, len(X_train), 100_000):
    for r in np.asarray(X_train[start:start + 100_000]):
        for i in test1_hash.get(row_hash(r), ()):
            duplicate[i] = True
print(f"\n{duplicate.sum()} test1 images have the same features as a training image")

for c in classifiers:
    rows.append(gap_row(c, "test1 without training duplicates", correct[c]["test1"][~duplicate], correct[c]["test2"]))
    rows.append(gap_row(c, "test1 training duplicates only", correct[c]["test1"][duplicate], correct[c]["test2"]))
overall = pd.DataFrame(rows)
overall.to_csv(OUT / "gen_overall_accuracy.csv", index=False)
print("\n=== Top-1 gap (test1 - test2), with 95% interval ===")
print(overall[["classifier", "subset", "test1_rows", "test1_top1", "test2_top1", "test1_top5", "test2_top5",
               "gap", "gap_ci_low", "gap_ci_high"]].round(4).to_string(index=False))


# 5. Per-class gap
# only 10 images per class in test set 2, so each class is tested (Fisher, FDR-corrected)
def per_class(clf):
    k1 = np.bincount(y["test1"], weights=correct[clf]["test1"], minlength=N_CLASSES)
    k2 = np.bincount(y["test2"], weights=correct[clf]["test2"], minlength=N_CLASSES)
    n1 = np.bincount(y["test1"], minlength=N_CLASSES)
    n2 = np.bincount(y["test2"], minlength=N_CLASSES)
    p = np.array([stats.fisher_exact([[k1[c], n1[c] - k1[c]], [k2[c], n2[c] - k2[c]]])[1] for c in range(N_CLASSES)])
    return pd.DataFrame({"class": np.arange(N_CLASSES), "name": class_name, "acc_test1": k1 / n1, "acc_test2": k2 / n2,
                         "gap": k1 / n1 - k2 / n2, "p_value": p, "q_value": stats.false_discovery_control(p)})


class_gaps = {c: per_class(c) for c in classifiers}
for c in classifiers:
    class_gaps[c].to_csv(OUT / f"gen_perclass_{c}.csv", index=False)
pc = class_gaps[PRIMARY]

# the per-class gaps sampling noise alone would give
rng = np.random.default_rng(SEED)
pooled = (pc.acc_test1 * 50 + pc.acc_test2 * 10) / 60
null_gaps = (rng.binomial(50, np.tile(pooled, 20)) / 50 - rng.binomial(10, np.tile(pooled, 20)) / 10)
print(f"\nPer-class gap ({NAMES[PRIMARY]}): sd {pc.gap.std():.3f} observed vs {null_gaps.std():.3f} expected from "
      f"sampling noise alone; {(pc.q_value < 0.05).sum()} of 1000 classes differ significantly (FDR 5%)")

worst = pc.sort_values(["q_value", "gap"], ascending=[True, False]).head(20)
print("\n=== 20 classes with the strongest evidence of a drop ===")
print(worst[["class", "name", "acc_test1", "acc_test2", "gap", "p_value", "q_value"]].round(3).to_string(index=False))


# 6. Error patterns
# the most common wrong predictions on test set 2, for the 10 classes above
errors = []
for c in classifiers:
    for k in worst["class"].head(10):
        wrong = (y["test2"] == k) & ~correct[c]["test2"]
        for pred, count in pd.Series(preds[c]["test2"][wrong, 0]).value_counts().head(3).items():
            errors.append({"classifier": NAMES[c], "true_class": k, "true_name": class_name[k],
                           "predicted_as": pred, "predicted_name": class_name[pred],
                           "count": count, "frac_of_errors": count / wrong.sum()})
pd.DataFrame(errors).to_csv(OUT / "gen_error_patterns.csv", index=False)
print("\nError patterns saved to gen_error_patterns.csv")


# 7. Class centroid shift
# baseline: distance between 10 and 40 test1 images of the same class
centroid_dist, baseline = np.zeros(N_CLASSES), np.zeros(N_CLASSES)
for k in range(N_CLASSES):
    A, B = X["test1"][y["test1"] == k], X["test2"][y["test2"] == k]
    centroid_dist[k] = np.linalg.norm(A.mean(axis=0) - B.mean(axis=0))
    splits = [rng.permutation(len(A)) for _ in range(NULL_SPLITS)]
    baseline[k] = np.mean([np.linalg.norm(A[s[:10]].mean(axis=0) - A[s[10:]].mean(axis=0)) for s in splits])
shift = pd.DataFrame({"class": np.arange(N_CLASSES), "name": class_name, "centroid_distance": centroid_dist,
                      "noise_baseline": baseline, "excess_shift": centroid_dist - baseline, "gap": pc.gap})
shift.sort_values("excess_shift", ascending=False).to_csv(OUT / "gen_feature_shift.csv", index=False)

print(f"\n=== Centroid shift: mean distance {centroid_dist.mean():.2f}, noise baseline {baseline.mean():.2f}; "
      f"larger than the baseline in {(centroid_dist > baseline).mean():.1%} of classes ===")
print(shift.sort_values("excess_shift", ascending=False).head(10).round(3).to_string(index=False))
for c in classifiers:
    g = class_gaps[c].gap
    print(f"{NAMES[c]}: correlation of excess shift with the class gap  Pearson {np.corrcoef(shift.excess_shift, g)[0, 1]:.3f}"
          f"  Spearman {stats.spearmanr(shift.excess_shift, g)[0]:.3f}")


# 8. Figures
main = overall[overall.subset == "all test1"]
fig, ax = plt.subplots(figsize=(7.5, 4))
x, w = np.arange(len(main)), 0.35
for offset, t, color, label in ((-w / 2, "test1", BLUE, "test set 1 (ImageNet val)"),
                                (w / 2, "test2", ORANGE, "test set 2 (ImageNetV2)")):
    acc = main[f"{t}_top1"].to_numpy()
    ax.bar(x + offset, acc, w, color=color, label=label, yerr=1.96 * np.sqrt(acc * (1 - acc) / n[t]), capsize=3)
    for i, a in enumerate(acc):
        ax.text(i + offset, a + 0.025, f"{100 * a:.1f}%", ha="center", fontsize=9)
ax.set_xticks(x, main.classifier)
ax.set_ylabel("Top-1 accuracy")
ax.set_ylim(0, 1.15)
ax.legend(frameon=False, loc="upper center", ncol=2)
ax.set_title("Generalisation gap: ImageNet val vs ImageNetV2")
ax.spines[["top", "right"]].set_visible(False)
plt.savefig(OUT / "fig_gen_overall.png", dpi=150, bbox_inches="tight")
plt.close()

fig, ax = plt.subplots(figsize=(6.5, 3.8))
bins = np.linspace(-1, 1, 41)
ax.hist(pc.gap, bins=bins, color=BLUE, alpha=0.85, edgecolor="white", label="observed")
ax.hist(null_gaps, bins=bins, weights=np.full(len(null_gaps), 1 / 20), histtype="step", color="k", lw=1.2,
        label="sampling noise only")
ax.axvline(0, color=GREY, lw=1)
ax.set_xlabel("Per-class top-1 gap (test1 - test2)")
ax.set_ylabel("Number of classes")
ax.set_title(f"Per-class gaps, {NAMES[PRIMARY]}")
ax.legend(frameon=False)
ax.spines[["top", "right"]].set_visible(False)
plt.savefig(OUT / "fig_gen_gap_hist.png", dpi=150, bbox_inches="tight")
plt.close()

fig, ax = plt.subplots(figsize=(6.2, 5))
significant = (pc.q_value < 0.05).to_numpy()
jitter = rng.uniform(-0.015, 0.015, N_CLASSES)       # gaps come in steps, so points would stack
ax.scatter(shift.excess_shift[~significant], pc.gap[~significant] + jitter[~significant], s=14, alpha=0.5, color=BLUE,
           label="not significant")
ax.scatter(shift.excess_shift[significant], pc.gap[significant] + jitter[significant], s=22, color=ORANGE,
           label="significant drop (FDR 5%)")
ax.axvline(0, color=GREY, lw=1)
ax.set_xlabel("Centroid shift beyond sampling noise (test1 vs test2)")
ax.set_ylabel(f"Per-class top-1 gap ({NAMES[PRIMARY]})")
ax.set_title("Feature shift vs generalisation gap")
ax.legend(frameon=False, loc="lower right")
ax.spines[["top", "right"]].set_visible(False)
plt.savefig(OUT / "fig_gen_shift_vs_gap.png", dpi=150, bbox_inches="tight")
plt.close()


# 9. Typicality
# cosine similarity of each test image to its own class centroid in train_small
def unit(Z):
    return Z / np.linalg.norm(Z, axis=1, keepdims=True)


X_small = np.load(PROC / "train_small_X.npy")
y_small = np.load(PROC / "train_small_y.npy").astype(int)
train_centroid = np.zeros((N_CLASSES, X_small.shape[1]))
np.add.at(train_centroid, y_small, unit(X_small))
train_centroid = unit(train_centroid)
typical = {t: (unit(X[t]) * train_centroid[y[t]]).sum(axis=1) for t in y}

edges = np.quantile(typical["test1"], np.linspace(0, 1, TYPICALITY_BINS + 1))[1:-1]
bin_of = {t: np.digitize(typical[t], edges) for t in y}
share = {t: np.bincount(bin_of[t], minlength=TYPICALITY_BINS) / n[t] for t in y}
typ_rows, explained = [], []
for c in classifiers:
    acc = {t: np.array([correct[c][t][bin_of[t] == b].mean() if (bin_of[t] == b).any() else np.nan
                        for b in range(TYPICALITY_BINS)]) for t in y}
    for b in range(TYPICALITY_BINS):
        typ_rows.append({"classifier": NAMES[c], "bin": b + 1, "share_test1": share["test1"][b],
                         "share_test2": share["test2"][b], "acc_test1": acc["test1"][b], "acc_test2": acc["test2"][b]})
    # test set 2's accuracy if only its mix of typical and atypical images differed
    expected2 = np.nansum(share["test2"] * acc["test1"])
    acc1, acc2 = correct[c]["test1"].mean(), correct[c]["test2"].mean()
    explained.append({"classifier": NAMES[c], "test1_top1": acc1, "test2_top1": acc2,
                      "test2_expected_from_typicality": expected2,
                      "share_of_gap_explained": (acc1 - expected2) / (acc1 - acc2)})
pd.DataFrame(typ_rows).to_csv(OUT / "gen_typicality_bins.csv", index=False)
explained = pd.DataFrame(explained)
explained.to_csv(OUT / "gen_typicality_explained.csv", index=False)
print(f"\n=== Typicality: mean similarity to own training centroid  test1 {typical['test1'].mean():.3f}"
      f"  test2 {typical['test2'].mean():.3f}; {share['test2'][0]:.1%} of test2 falls in test1's least typical 10% ===")
print(explained.round(4).to_string(index=False))

typ = pd.DataFrame(typ_rows)
typ = typ[typ.classifier == NAMES[PRIMARY]]
fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
sim_bins = np.linspace(min(typical["test2"].min(), typical["test1"].min()), 1, 50)
for t, color, label in (("test1", BLUE, "test set 1"), ("test2", ORANGE, "test set 2")):
    axes[0].hist(typical[t], bins=sim_bins, density=True, histtype="step", lw=1.6, color=color,
                 label=f"{label} (mean {typical[t].mean():.3f})")
    axes[1].plot(typ.bin, typ[f"acc_{t}"], "o-", color=color, label=label)
axes[0].set_xlabel("Cosine similarity to own-class training centroid")
axes[0].set_ylabel("Density")
axes[0].set_title("Typicality of test images")
axes[1].set_xlabel("Typicality decile of test set 1 (1 = least typical)")
axes[1].set_ylabel(f"Top-1 accuracy ({NAMES[PRIMARY]})")
axes[1].set_title("Accuracy at equal typicality")
axes[1].set_xticks(range(1, TYPICALITY_BINS + 1))
for ax in axes:
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
plt.savefig(OUT / "fig_gen_typicality.png", dpi=150, bbox_inches="tight")
plt.close()


# 10. Per feature
# Welch t-test and Cohen's d per feature; both sets have the same class mix
X1, X2 = X["test1"], X["test2"]
_, p_feat = stats.ttest_ind(X2, X1, axis=0, equal_var=False)
d = (X2.mean(axis=0) - X1.mean(axis=0)) / np.sqrt((X1.var(axis=0, ddof=1) + X2.var(axis=0, ddof=1)) / 2)


# separability = share of variance between class means; test1 cut to 10 per class to match test2
def eta_squared(Z, labels):
    counts = np.bincount(labels, minlength=N_CLASSES)
    means = np.zeros((N_CLASSES, Z.shape[1]))
    np.add.at(means, labels, Z)
    means /= counts[:, None]
    between = (counts[:, None] * (means - Z.mean(axis=0)) ** 2).sum(axis=0) / len(Z)
    return between / Z.var(axis=0)


eta_train = eta_squared(X_small, y_small)
eta_test2 = eta_squared(X2, y["test2"])
eta_test1 = np.zeros(X1.shape[1])
for _ in range(ETA_REPEATS):
    rows = np.concatenate([rng.choice(np.flatnonzero(y["test1"] == k), 10, replace=False) for k in range(N_CLASSES)])
    eta_test1 += eta_squared(X1[rows], y["test1"][rows]) / ETA_REPEATS
features = pd.DataFrame({"feature": np.arange(X1.shape[1]), "mean_test1": X1.mean(axis=0), "mean_test2": X2.mean(axis=0),
                         "cohens_d": d, "p_value": p_feat, "q_value": stats.false_discovery_control(p_feat),
                         "eta2_train": eta_train, "eta2_test1": eta_test1, "eta2_test2": eta_test2,
                         "eta2_drop": eta_test1 - eta_test2})
features.sort_values("eta2_drop", ascending=False).to_csv(OUT / "gen_feature_dimensions.csv", index=False)

n_sig = (features.q_value < 0.05).sum()
print(f"\n=== Per feature: {n_sig} of {len(features)} features shift significantly (FDR 5%); "
      f"|d| median {np.median(np.abs(d)):.3f}, max {np.abs(d).max():.3f} ===")
print(f"separability (eta squared, 10 per class): test1 mean {eta_test1.mean():.3f}, test2 mean {eta_test2.mean():.3f}; "
      f"lower on test2 for {(eta_test2 < eta_test1).mean():.1%} of features")
print(f"Spearman: |d| vs training separability {stats.spearmanr(np.abs(d), eta_train)[0]:.3f}; "
      f"separability drop vs training separability {stats.spearmanr(features.eta2_drop, eta_train)[0]:.3f}")
print(features.sort_values("eta2_drop", ascending=False).head(10)
      [["feature", "cohens_d", "q_value", "eta2_train", "eta2_test1", "eta2_test2", "eta2_drop"]].round(3).to_string(index=False))

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].hist(d, bins=50, color=BLUE, edgecolor="white")
axes[0].axvline(0, color=GREY, lw=1)
axes[0].set_xlabel("Cohen's d per feature (test2 - test1)")
axes[0].set_ylabel("Number of features")
axes[0].set_title(f"Mean shift: {n_sig} of {len(d)} features significant")
lims = [0, max(eta_test1.max(), eta_test2.max()) * 1.05]
sc = axes[1].scatter(eta_test1, eta_test2, c=eta_train, s=10, cmap="viridis")
axes[1].plot(lims, lims, "--", color=GREY)
axes[1].set_xlim(lims)
axes[1].set_ylim(lims)
axes[1].set_xlabel("Class separability on test set 1 (eta squared)")
axes[1].set_ylabel("Class separability on test set 2")
axes[1].set_title("Class separability per feature")
fig.colorbar(sc, ax=axes[1], label="separability on training data")
for ax in axes:
    ax.spines[["top", "right"]].set_visible(False)
plt.savefig(OUT / "fig_gen_feature_dimensions.png", dpi=150, bbox_inches="tight")
plt.close()


# 11. Clustering
# k-means fitted on train_small; ~200 test set 2 images per cluster, enough to test a gap
pca = PCA(n_components=PCA_DIMS, random_state=SEED).fit(X_small)
kmeans = KMeans(n_clusters=N_CLUSTERS, n_init=3, random_state=SEED).fit(pca.transform(X_small))
cluster = {t: kmeans.predict(pca.transform(X[t])) for t in y}
train_cluster = kmeans.labels_

cl_rows = []
for k in range(N_CLUSTERS):
    m1, m2 = cluster["test1"] == k, cluster["test2"] == k
    top_classes = pd.Series(y_small[train_cluster == k]).value_counts().head(3).index
    a1, a2 = correct[PRIMARY]["test1"][m1], correct[PRIMARY]["test2"][m2]
    p = stats.fisher_exact([[a1.sum(), (~a1).sum()], [a2.sum(), (~a2).sum()]])[1] if m2.any() else np.nan
    cl_rows.append({"cluster": k, "example_classes": ", ".join(class_name[top_classes]),
                    "share_test1": m1.mean(), "share_test2": m2.mean(), "over_representation": m2.mean() / m1.mean(),
                    "acc_test1": a1.mean(), "acc_test2": a2.mean() if m2.any() else np.nan,
                    "gap": a1.mean() - a2.mean() if m2.any() else np.nan, "p_value": p,
                    "gap_contribution": (a1.sum() / n["test1"] - a2.sum() / n["test2"])})
clusters = pd.DataFrame(cl_rows)
clusters["q_value"] = stats.false_discovery_control(clusters.p_value.fillna(1))
clusters["gap_contribution"] /= clusters.gap_contribution.sum()      # share of the overall gap
clusters.sort_values("gap", ascending=False).to_csv(OUT / "gen_clusters.csv", index=False)
table = np.vstack([np.bincount(cluster[t], minlength=N_CLUSTERS) for t in y])
chi2, p_chi2 = stats.chi2_contingency(table[:, table.sum(axis=0) > 0])[:2]
print(f"\n=== Clusters ({N_CLUSTERS} k-means clusters fitted on train_small): test set membership differs, "
      f"chi-squared {chi2:.0f}, p = {p_chi2:.2g}; {(clusters.q_value < 0.05).sum()} clusters have a significant gap ===")
print(f"Spearman: cluster over-representation in test2 vs cluster gap {stats.spearmanr(clusters.over_representation, clusters.gap, nan_policy='omit')[0]:.3f}")
print(clusters.sort_values("gap", ascending=False).head(10)
      [["cluster", "example_classes", "share_test1", "share_test2", "acc_test1", "acc_test2", "gap", "q_value"]]
      .round(3).to_string(index=False))

fig, ax = plt.subplots(figsize=(6.5, 4.5))
sig = clusters.q_value < 0.05
ax.scatter(clusters.acc_test1[~sig], clusters.acc_test2[~sig], s=4000 * clusters.share_test2[~sig], alpha=0.5,
           color=BLUE, label="gap not significant")
ax.scatter(clusters.acc_test1[sig], clusters.acc_test2[sig], s=4000 * clusters.share_test2[sig], alpha=0.8,
           color=ORANGE, label="significant gap (FDR 5%)")
lo = min(clusters.acc_test1.min(), clusters.acc_test2.min()) - 0.03
ax.plot([lo, 1], [lo, 1], "--", color=GREY)
for _, r in clusters.sort_values("gap", ascending=False).head(4).iterrows():
    ax.annotate(r.example_classes.split(",")[0], (r.acc_test1, r.acc_test2), fontsize=8, xytext=(4, -10),
                textcoords="offset points")
ax.set_xlabel("Top-1 on test set 1")
ax.set_ylabel("Top-1 on test set 2")
ax.set_title(f"k-means clusters ({NAMES[PRIMARY]}); size = share of test set 2")
ax.legend(frameon=False, loc="upper left", markerscale=0.3)
ax.spines[["top", "right"]].set_visible(False)
plt.savefig(OUT / "fig_gen_clusters.png", dpi=150, bbox_inches="tight")
plt.close()


# 12. Misclassified ImageNetV2 images
v2_dir = args.v2_images
meta2 = pd.read_csv(PROC / "test2_meta.csv")
if (v2_dir / meta2.path[0]).exists():
    show = worst["class"].head(6).to_numpy()
    fig, axes = plt.subplots(len(show), 5, figsize=(12, 2.6 * len(show)))
    for row, k in zip(axes, show):
        wrong = np.flatnonzero((y["test2"] == k) & ~correct[PRIMARY]["test2"])[:5]
        for ax in row:
            ax.axis("off")
        for ax, i in zip(row, wrong):
            ax.imshow(Image.open(v2_dir / meta2.path[i]).convert("RGB"))
            ax.set_title(f"predicted: {class_name[preds[PRIMARY]['test2'][i, 0]]}", fontsize=8)
        row[0].text(-0.08, 0.5, class_name[k], transform=row[0].transAxes, ha="right", va="center", fontsize=10,
                    weight="bold")
    fig.tight_layout()
    fig.suptitle(f"Misclassified ImageNetV2 images ({NAMES[PRIMARY]}), classes with the strongest drop", y=1.01)
    plt.savefig(OUT / "fig_gen_v2_errors.png", dpi=110, bbox_inches="tight")
    plt.close()
    print("\nImage figure saved: fig_gen_v2_errors.png")
else:
    print(f"\nImageNetV2 images not found under {v2_dir}; skipping the image figure (pass --v2-images)")

print("\nTables and figures saved to data/output/: gen_*.csv and fig_gen_*.png")
