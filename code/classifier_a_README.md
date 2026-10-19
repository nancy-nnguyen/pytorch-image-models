# Classifier A: regularised linear classification

## Method

Classifier A uses the supplied frozen EVA-02 features: 1,024 numerical
values per image across 1,000 classes.

The classifier uses scikit-learn's SGDClassifier with logistic loss,
an L2 penalty, and one-versus-rest multiclass classification.

StandardScaler is fitted exclusively on training examples and applied
unchanged to validation and test examples. Full-data scaling and training
are performed incrementally in batches of 1,024.

The shared internal validation set is used for configuration selection
and early stopping. Neither test set is used for fitting or tuning.

All experiments use seed 7.

## Configuration selection

Three alpha values were compared on the shared 100,000-example training
subset, containing 100 examples per class, for five epochs each.

| Alpha | Epoch-5 validation top-1 | Epoch-5 validation top-5 |
|---|---:|---:|
| 0.00001 | 95.014% | 99.646% |
| 0.0001 | 94.924% | 99.660% |
| 0.001 | 94.730% | 99.622% |

Alpha=0.00001 achieved the highest validation top-1 accuracy and was
selected for further training.

The default learning-rate schedule also depends on alpha. These
experiments therefore compare complete training configurations rather
than isolating the effect of regularisation.

A ten-epoch subset run reached its highest validation top-1 accuracy
of 95.054% at epoch 8.

The tuning runs use patience equal to the requested epoch limit so
that every requested epoch is recorded.

## Full training

The full feature array is memory-mapped. Only rows selected by
split_train_idx.npy are used for fitting, excluding the shared internal
validation examples.

Training uses batches of 1,024, a maximum of ten epochs, and early
stopping after three consecutive epochs without improved validation
top-1 accuracy. The best model and fitted scaler are saved together.

The selected checkpoint was epoch 2:

- Internal validation top-1: 95.228%.
- Internal validation top-5: 99.656%.
- Training examples: 1,231,167.
- Training stopped after epoch 5.
- Original run: approximately 36.8 minutes including scaling and validation.
- Reproduction run: approximately 62.43 minutes across training/validation
  epochs, excluding initial scaler fitting.

The regenerated tuning and full-training accuracy results matched the
original results. Execution times differed between runs.

The original run emitted scikit-learn parallel configuration warnings
but completed. Software versions are recorded in
data/output/classifier_a_environment.txt.

## Final test results

The saved epoch-2 checkpoint was evaluated without further fitting or
configuration changes.

| Dataset | Examples | Top-1 | Top-5 |
|---|---:|---:|---:|
| ImageNet: test1 | 50,000 | 89.856% | 98.988% |
| ImageNetV2 MatchedFrequency: test2 | 10,000 | 82.280% | 96.390% |

The top-1 gap is 7.576 percentage points.
The top-5 gap is 2.598 percentage points.

These test results were unchanged in the reproduction run.

Model choice must be based on internal validation performance, not on
which classifier achieves the highest test1 or test2 score.

## Decision scores

Exported decision scores and margins are not calibrated probabilities.

Do not use decision_margin or top*_score as confidence probabilities,
as inputs to probability-based confidence histograms, or as quantities
directly comparable with MLP softmax outputs.

If the group uses MLP softmax outputs for confidence analysis, it should
also acknowledge that softmax outputs are not necessarily calibrated.

## Per-class analysis

ImageNetV2 top-1 accuracy was lower for 560 classes, equal for 113 classes,
and higher for 327 classes.

Both test sets contain all 1,000 classes and are balanced within each set:

- ImageNet: 50 examples per class.
- ImageNetV2: 10 examples per class.

Individual class differences are descriptive and subject to sampling
variation. One ImageNetV2 prediction changes that class's accuracy by
10 percentage points.

## Exact-feature overlap sensitivity analysis

Of the 50,000 test1 examples, 773 had feature vectors exactly matching
at least one example in the classifier's training-only split.

Hash matches were verified through element-wise feature equality.

| Test1 group | Examples | Top-1 accuracy |
|---|---:|---:|
| Complete test set | 50,000 | 89.856% |
| Exact training-feature match | 773 | 35.317% |
| No exact training-feature match | 49,227 | 90.712% |

Excluding matching examples increased the test1-test2 top-1 gap from
7.576 to approximately 8.432 percentage points. Thus, these matches
do not explain an inflated test1 score for this classifier.

A follow-up check found that all 773 matching test1 examples had a
different label from their first exact-matching training row. The
classifier predicted that training row's label for 439 examples (56.8%).

This identifies conflicting labels associated with identical supplied
feature vectors. It does not establish which labels are correct,
whether the original image files are identical, or the source of the
discrepancy. Where multiple training matches exist, only the first
matching row was checked for label agreement.

The original complete-test-set scores are retained. The non-matching
subset result is reported separately as a descriptive sensitivity
analysis, not a replacement benchmark. Excluding matching examples
also changes the composition of the evaluated subset.

No test-informed relabelling or removal of training examples was
performed.

The 773 count refers specifically to matches against the actual
training-only split. Counts obtained against the entire original
training array should be reported separately.

## Reproduction

Use Python 3.11 and install the assignment dependencies:

```powershell
python -m pip install -r A3_requirements.txt
```

Run the following commands from the repository root, with the shared
prepared arrays and metadata in data/processed/.

The commands explicitly overwrite each run's existing history and
checkpoint. Back up previous outputs first if they should be retained.

### Regenerate tuning histories

```powershell
python code/classifier_a_linear.py --per-class 100 --epochs 5 --patience 5 --alpha 0.00001 --overwrite
python code/classifier_a_linear.py --per-class 100 --epochs 5 --patience 5 --alpha 0.0001 --overwrite
python code/classifier_a_linear.py --per-class 100 --epochs 5 --patience 5 --alpha 0.001 --overwrite
python code/classifier_a_linear.py --per-class 100 --epochs 10 --patience 10 --alpha 0.00001 --overwrite
```

### Train the full-data model

```powershell
python code/classifier_a_linear.py --full --epochs 10 --patience 3 --alpha 0.00001 --overwrite
```

### Evaluate and analyse

```powershell
python code/classifier_a_evaluate.py
python code/classifier_a_analyse.py
python code/classifier_a_check_overlap_labels.py
```

The analysis script requires the tuning histories produced above and
the prediction files produced by the evaluation script. Run the
overlap-label check after the analysis script.

The evaluation script loads:

```text
data/models/classifier_a_full_alpha1e-05_epochs10_best.joblib
```

The --overwrite flag is required when reproducing training in a checkout
that already contains the committed histories. Without it, training
refuses to replace an existing history or checkpoint.

## Scripts and outputs

| Script | Purpose |
|---|---|
| code/classifier_a_linear.py | Subset and full-data training, validation histories, and best-checkpoint saving |
| code/classifier_a_evaluate.py | Frozen-checkpoint test evaluation and per-image prediction export |
| code/classifier_a_analyse.py | Figures, per-class comparison, and exact-feature overlap analysis |
| code/classifier_a_check_overlap_labels.py | Label comparison against each test example's first matching training row |

Subset training histories use:

```text
data/output/classifier_a_dev_n100_alpha*_epochs*.csv
```

The history schema includes mode and training_rows. Both subset and
full-data histories are generated by the live training implementation.

Results, predictions, analysis tables, and figures are written to
data/output/. Model checkpoints are written to data/models/.

The three report figures are:

- fig_classifier_a_learning.png
- fig_classifier_a_test_accuracy.png
- fig_classifier_a_class_gaps.png

Datasets and trained model files are excluded from Git. Selected result
CSVs, figures, and the environment record are committed for review.

## Limitations

Configuration selection used one shared validation split and one random
seed. Variability across alternative splits or seeds was not measured.

The overlap analysis identifies a supplied-data concern requiring further
investigation, but does not establish its cause.

This contribution provides the baseline classifier, test evaluation,
and descriptive analyses. A hypothesis-driven classifier refinement
remains a separate group experiment.