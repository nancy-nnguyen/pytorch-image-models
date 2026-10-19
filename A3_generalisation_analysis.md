# Part (b): why accuracy drops on ImageNetV2

*Generalisation analysis (workstream 5). Script: `code/05_generalisation_analysis.py`. All numbers below come from its run on the predictions of classifiers A, B and C. Tables are `data/output/gen_*.csv`; figures are `data/output/fig_gen_*.png`.*

## The question

All three classifiers on the frozen EVA-02 features lose between 7 and 9 points of top-1 accuracy when moving from test set 1 (ImageNet val) to test set 2 (ImageNetV2 MatchedFrequency). Recht et al. (2019) built ImageNetV2 to repeat ImageNet's collection process as closely as possible. They found the same kind of drop for every model they tested, and no evidence that the drop came from the community over-tuning to the original validation set. Instead, they attributed it to small differences in how the new images were collected and filtered. This analysis asks what the drop looks like in our features: is it a few classes, a few features, a new kind of image, or something spread across everything?

Nothing in this analysis trains or tunes a classifier. Every centroid, PCA and clustering model below is fitted on `train_small` (100 training images per class), and the test sets are only read.

## 1. The gap is real, and slightly larger than it first looks

| Classifier | Test set 1 | Test set 2 | Gap (95% CI) | Gap without the 798 duplicates |
|---|---|---|---|---|
| A: regularised linear | 89.9% | 82.3% | 7.6 (6.8–8.4) | 8.5 |
| B: small MLP | 89.4% | 80.7% | 8.8 (8.0–9.6) | 9.8 |
| C: nearest centroid | 89.9% | 82.7% | 7.3 (6.5–8.0) | 8.1 |

The interval treats the two test sets as independent samples, and none of them comes close to zero (Figure `fig_gen_overall.png`). The top-5 gap is much smaller (A: 99.0% vs 96.4%, a 2.6-point gap), which already suggests that most errors on ImageNetV2 put the true class close behind the predicted one.

Test set 1 contains 798 images whose 1,024 features exactly match a training image. Identical features mean identical input images. In every case the training copy carries a *different* label, usually a neighbouring class such as tiger shark vs great white shark. This is consistent with ImageNet having removed duplicates only within a class. These images are label-ambiguous, and the classifiers get only 29–40% of them right. Because they pull test set 1 down, excluding them *widens* the gap by about one point. ImageNetV2 has no such overlap.

## 2. It is not a handful of classes

With only 10 ImageNetV2 images per class, one image moves a class's accuracy by 10 points, so a ranking of the "worst classes" is mostly noise. Each class was given Fisher's exact test (50 test set 1 images vs 10 test set 2 images), with a Benjamini–Hochberg false-discovery-rate correction across the 1,000 classes.

Per-class gaps do spread more widely than noise alone would produce (standard deviation 0.17 against 0.10 expected; `fig_gen_gap_hist.png`). But only **27 of the 1,000 classes** show a significant drop for classifier A, among them soup bowl, silky terrier, toaster, limpkin, dining table, church and coffee mug. The rest of the gap is a small loss spread over many classes, each too small to detect at 10 images.

Grouping classes gives enough images to see this. k-means with 50 clusters, fitted on the training features (Lab 3), places about 200 ImageNetV2 images in each cluster. **27 of the 50 clusters** show a significant drop, together accounting for 95% of the gap (`fig_gen_clusters.png`). The steepest drops are in fine-grained dog breeds (Cardigan/Pembroke corgis, the Yorkshire/silky/Australian terriers, poodles) and in shop and food scenes. But the largest *contributions* to the gap come from big mixed clusters where the drop per image is modest.

## 3. ImageNetV2 holds the same kinds of images

If ImageNetV2 contained different content (new kinds of scenes, or objects that ImageNet lacks), its images would fall into the clusters in different proportions. They do not: the cluster membership of the two test sets is statistically indistinguishable (chi-squared test, p = 0.3). The shift happens *within* each region of feature space, not between regions.

## 4. Within each class, ImageNetV2 images sit further from the training data

For every class, the centroid of its ImageNetV2 images lies further from the centroid of its test set 1 images (mean distance 15.5) than two samples of the same test set 1 class do from each other (mean 9.6, from random 10-vs-40 splits). This holds in 98% of classes. Classes whose centroid moves furthest beyond that noise baseline lose the most accuracy (Pearson 0.53, Spearman 0.50, nearly identical for all three classifiers; `fig_gen_shift_vs_gap.png`). The classifiers fail where EVA-02 sees ImageNetV2 images as different from that class's training images.

## 5. It is not a few bad features

Per feature, a Welch's t-test between the test sets finds 729 of the 1,024 features shifted significantly. However, every shift is small (median |Cohen's d| 0.04, largest 0.40; `fig_gen_feature_dimensions.png`, left).

To see which features "work for one test set but not the other", each feature's class separability was measured. This is the share of its variance that lies between class means (eta squared). Test set 1 was subsampled to 10 images per class so that both sets are compared at the same sample size. Separability is lower on ImageNetV2 for **every one** of the 1,024 features (mean 0.652 → 0.574; `fig_gen_feature_dimensions.png`, right). The size of the drop is unrelated to how useful the feature is on training data (Spearman 0.04), and so is the size of the mean shift (−0.06). There is no small set of features that breaks on ImageNetV2. Each class's ImageNetV2 images are simply more spread out, along all features at once.

## 6. The gap is explained by typicality

The simplest summary of points 4 and 5 is *typicality*: the cosine similarity of an image's features to the training centroid of its own class. ImageNetV2 images are less typical on average (0.705 vs 0.785). 20.8% of them fall in the least typical tenth of test set 1, twice the expected share (`fig_gen_typicality.png`, left).

The decisive comparison holds typicality fixed. Within each typicality decile of test set 1, classifier A is *as accurate* on ImageNetV2 as on ImageNet val, and sometimes slightly more: 30% vs 31% in the least typical tenth, 84% vs 81% in the next, and above 93% in both from the third tenth on (`fig_gen_typicality.png`, right).

So the accuracy ImageNetV2 *would* have, if only its mix of typical and atypical images differed, can be predicted from test set 1 alone: test set 1's per-decile accuracy weighted by ImageNetV2's share of each decile. That prediction accounts for essentially the whole gap:

| Classifier | Test set 2 actual | Predicted from typicality | Share of gap explained |
|---|---|---|---|
| A | 82.3% | 81.8% | 106% |
| B | 80.7% | 81.4% | 91% |
| C | 82.7% | 81.8% | 112% |

Shares above 100% are within the noise of a 10,000-image test set; read them as "all of it".

## 7. What the atypical images look like

`fig_gen_v2_errors.png` shows the misclassified ImageNetV2 images of the six classes with the strongest evidence of a drop. Almost none of the errors are absurd. They fall into a few kinds:

- **More than one valid label.** Most "soup bowl" errors are predicted as consommé, and the bowls do contain soup. Errors on "web site" are screenshots of products, predicted as the product shown (modem, book jacket).
- **Fine-grained siblings.** Silky terrier is confused with Australian and Yorkshire terrier, Welsh springer spaniel with English springer, and limpkin with black stork and herons.
- **Object small or in context.** A silky terrier small in a garden is predicted as "park bench", and a spaniel beside a train is predicted correctly as a springer breed but the wrong one.
- **A different variant of the object.** Three of the five "toaster" errors shown are toaster ovens, predicted as rotisserie or microwave.

The same pattern shows in the error table (`gen_error_patterns.csv`) and in the small top-5 gap.

## Hypothesis

> **ImageNetV2 is harder because it contains a larger share of atypical images. These include ambiguous multi-object scenes, fine-grained look-alikes, small or unusual views, and different variants of the object. It does not contain new kinds of images, and no particular features fail on it. At equal typicality, all three classifiers are as accurate on ImageNetV2 as on ImageNet val.**

This agrees with Recht et al.'s conclusion that the drop comes from the data-collection process making the new images slightly harder, not from adaptive overfitting. It adds that, in EVA-02's feature space, the difference is a widening of every class's distribution rather than a move to new regions.

**Caveat.** Typicality is measured against the image's *given* label. An image whose label is wrong also looks atypical, so the features alone cannot separate "harder images" from "noisier labels". The image figure suggests the former dominates: most errors are defensible readings of an ambiguous image, not mislabelled photographs.

## What this predicts for part (c)

Each prediction can be checked on our own validation set before touching a test set. The validation set has its own least-typical tenth, built the same way.

1. **Dropping or down-weighting the most shifted features should not narrow the gap**, because the shift is spread over all features and unrelated to their importance. This is a cheap test, and a negative result is a valid answer.
2. **What helps is anything that improves accuracy on atypical images.** Test set 2 has twice test set 1's share of them, so a gain there narrows the gap. Candidates are re-weighting training images by how atypical they are within their class, and combining the three classifiers, which make partly different errors. The yardstick is accuracy on the least typical tenth of our validation set.
3. **A head on frozen features has a ceiling.** Points 5 and 6 say the information that separates these images is weaker in EVA-02's features themselves, so we should expect the gap to narrow only partly. The spec accepts that, provided the attempt is reported.

## Not covered yet

- **Prediction confidence per test set.** This needs the MLP to save its softmax scores, because classifier A's decision scores are not probabilities. Typicality covers much of the same ground.
- **Test set 1 images.** They are not stored locally (only the ImageNetV2 images are), so the visual comparison uses ImageNetV2 errors only.

## Reproducing

From the repository root, after the three classifier scripts have written their predictions:

    python code/05_generalisation_analysis.py --v2-images <folder that holds imagenetv2-matched-frequency-format-val>

It runs in about 1.5 minutes on a laptop CPU. Without `--v2-images` it skips only the image figure.

## Reference

Recht, B., Roelofs, R., Schmidt, L. and Shankar, V. (2019). Do ImageNet classifiers generalize to ImageNet? *Proceedings of the 36th International Conference on Machine Learning*, PMLR 97, pp. 5389–5400.
