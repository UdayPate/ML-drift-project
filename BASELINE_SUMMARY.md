# Initial Logistic Regression baseline: summary for the mid-semester presentation

> **Preliminary development-set results, not final held-out stream results.** The reserved stream rows (last 20%) have not been used, inspected or scored. This is one split, one untuned model and one fixed threshold.
> Source: `notebooks/03_baseline_logreg.ipynb` (executed from a fresh kernel, 0 errors). Figures: `figures/07_baseline_confusion_matrix.png`, `figures/08_baseline_pr_curve.png`.

## Setup

- **Chronological split (60/20/20, original row order, no shuffling):** train = rows 0 to 24,711 (24,712 rows, 1,188 yes, 4.81%); development = rows 24,712 to 32,949 (8,238 rows, 912 yes, 11.07%); reserved stream = rows 32,950 to 41,187 (8,238 rows, untouched).
- **Conservative 10-feature baseline:** `age`, `job`, `marital`, `education`, `default`, `housing`, `loan`, `pdays`, `previous`, `poutcome`. Excluded: `duration` (known only after the call), the current-contact features (`campaign`, `contact`, `month`, `day_of_week`) and the five economic indicators, because their availability before the call is unresolved (`notebooks/02_leakage_review.ipynb`).
- **Leakage-safe preprocessing:** one scikit-learn pipeline fitted on **training rows only**: `age` standardised; `pdays = 999` treated as a sentinel (indicator `pdays_is_999` plus the real value, median-imputed from training rows, not standardised); `previous` unscaled; categoricals one-hot encoded with `"unknown"` as its own category. Development and stream rows are only transformed, never used to fit. A unit test confirms that changing every later row leaves the fitted preprocessing identical.
- **Model:** one Logistic Regression with library defaults (L2, `C = 1.0`), no class weights, no tuning, no threshold optimisation, seed 42. Positive class = subscribed ("yes"). Threshold-based metrics use 0.5, a documented starting point and not a tuned choice.

## Development-set results (8,238 rows)

| Metric | Value |
|---|---|
| Development positive rate | 0.1107 (11.07%) |
| **PR-AUC (average precision)** | **0.1726** |
| PR-AUC of random ranking (reference) | 0.1107, so the model is 1.56x the reference |
| ROC-AUC | 0.5819 (0.5 = random ranking) |
| Predicted "yes" at threshold 0.5 | **0 of 8,238 (0.00%)** |
| Precision / recall / F1 at 0.5 | 0 / 0 / 0 (precision is reported as 0 when nothing is predicted positive) |

Confusion matrix (threshold 0.5, positive class = subscribed):

| | Predicted: no | Predicted: yes |
|---|---|---|
| **Actual: no** | 7,326 (true negatives) | 0 (false positives) |
| **Actual: yes** | 912 (false negatives) | 0 (true positives) |

In-sample sanity check on the training rows (not evidence of generalisation): PR-AUC 0.0577 vs. reference 0.0481, ROC-AUC 0.5534.

## Why does the model predict zero positives at 0.5?

The highest predicted probability on the development rows is **0.425**, which is below 0.5, so no development row is classified as positive at that threshold. Logistic Regression's probabilities reflect the positive rate it was trained on (4.81%); the development rows have a higher rate (11.07%), but the model's probabilities stay low. These are explanations supported by descriptive facts, not proven causes. In the training rows the history features barely vary (99.47% of rows have `poutcome = nonexistent`; only 130 rows have `previous > 0`, versus 26.04% of development rows), and even on the rows it was fitted on the model separates yes from no only weakly (in-sample ROC-AUC 0.553).

## Weak ranking vs. zero threshold-based recall: two different things

- **Ranking (PR-AUC, ROC-AUC) does not depend on a threshold.** It asks whether real "yes" rows tend to get higher probabilities than "no" rows. Here they do, but only slightly: PR-AUC 0.1726 vs. 0.1107 for random ranking, ROC-AUC 0.5819. The precision-recall curve (`figures/08_baseline_pr_curve.png`) sits above the random-ranking line over most of the recall range and converges to it at high recall; for recall above about 0.94 it dips marginally below the line (by at most about 0.002 in precision).
- **Threshold metrics (precision, recall, F1, the confusion matrix) depend on a cut-off.** A model can rank somewhat better than random and still never reach 0.5. Zero recall at 0.5 is therefore a statement about the cut-off, not proof that the ranking carries no information. A different cut-off would be a new decision; the project-wide threshold choice is still pending, and we did not tune it here.

## What this does not show

- **It does not show that drift caused performance degradation.** The training rows and development rows differ (positive rate 4.81% vs. 11.07%; history features rare in training), which is consistent with change over time. But the model is already weak on the rows it was trained on, we have a single development window, no comparison across time, and no retraining baseline. The weak result cannot be attributed to drift.
- **It says nothing yet about retraining strategies.** Static, periodic and drift-triggered retraining have not been implemented or evaluated.
- **It is not a final result.** It is a preliminary baseline for the later experiments.

## Where each number comes from (`notebooks/03_baseline_logreg.ipynb`)

| Number(s) | Notebook section |
|---|---|
| Split sizes, yes counts, positive rates 4.81% / 11.07% | 1. Data and chronological split |
| Prevalence, PR-AUC, reference, ROC-AUC, threshold metrics, predicted-positive count | 3. Development-set results |
| Confusion matrix | 3. Confusion matrix (figure 07) |
| PR curve and agreement of the curve's step-sum average precision with 0.1726 | 3. Precision-recall curve (figure 08) |
| In-sample PR-AUC 0.0577 / reference 0.0481 / ROC-AUC 0.5534 | 5. In-sample sanity check |
| Highest predicted probability 0.425; 99.47%, 130 rows, 26.04% | 6 and 7. Diagnostics and investigation |
