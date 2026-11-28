# Retraining strategies and the full run: method note and handoff

Person 3's part of the final project. Code: `src/strategies.py`. Run: `notebooks/06_experiment.ipynb` → `results/results.csv`. Tests: `tests/test_strategies.py` (17 tests, no dataset needed). All settings are in `src/settings.py` and are **PROPOSED, pending team approval** (see `DECISIONS.md`).

> Numbers below come from `notebooks/06_experiment.ipynb` run top to bottom on the real `bank-additional-full.csv` on 2026-11-26. All sanity checks in the notebook passed. These are first results under PROPOSED settings; the statistical comparison is Person 4's.

## For the report (method note)

**Strategies.** We compare three ways of maintaining a classifier on a stream of 8 windows. The *static* model is fitted once on the initial training rows and never changed. The *periodic* model is retrained before every second window (windows 3, 5 and 7). The *drift-triggered* model is retrained when the drift rule from the monitoring step fires on the incoming window, i.e. when any model-input feature has PSI above 0.25 against the rows the current model was trained on.

**One shared loop (fair comparison).** All three strategies run through the same function. For a given scenario, model and seed they receive the identical stream, the same model configuration and features, the same label-delay and retraining-data rules, the same threshold rule, and they start from one shared initial model. The only difference between them is the rule that decides whether to retrain before a window. A unit test checks that the three strategies produce identical results on every window before the first retrain.

**Timeline of a window.** When window *i* arrives its features are visible but its labels are not. Drift monitoring compares the window's features with the current model's training rows; this is logged for every strategy, though only the drift-triggered strategy acts on it. A retrain may only use labelled rows: the initial training rows plus the stream windows whose labels have arrived. With a label delay of one window, before window *i* the windows up to *i* − 2 are labelled; the current and future windows are never used. If no new labelled window has arrived since the last fit, the retrain is skipped and logged. The window is then scored and evaluated against its true labels.

**Retraining data and threshold.** Retraining uses an expanding window: all labelled rows available so far. Preprocessing and model are refitted together on exactly those rows. The decision threshold is re-tuned at every fit, including the initial one: the most recent 1,030 labelled rows (one window) are held out, a temporary model is fitted on the older rows, the cut-off with the best F1 on the held-out rows is chosen, and the model is then refitted on all labelled rows with that cut-off. If the held-out rows contain fewer than 10 "yes" rows, the previous cut-off is kept. Hyperparameters stay fixed at Person 2's selected configuration throughout the stream.

**Models and grid.** Person 2's shortlist: Logistic Regression and Linear SVM on the 10 baseline features (adding the economic indicators lowered development PR-AUC for every model). Linear SVM produces margin scores, not probabilities; PR-AUC and the F1-tuned cut-off work on either. The full run is 4 scenarios × 3 strategies × 2 models × 3 seeds (42, 43, 44) × 8 windows = 576 window rows. The real replay does not depend on the seed and both models are deterministic, so its three seeds give identical rows.

**What is logged per window.** PR-AUC and the random-guess PR-AUC (the window's share of "yes"), ROC-AUC, F1, precision, recall, the confusion counts and the threshold used; whether the drift rule fired and which features fired; the largest PSI among model inputs and among economic indicators (economic alerts are logged separately and never trigger); whether a retrain was wanted, done, or skipped for lack of new labels; retrain time; the number of rows the current model was trained on; and drift-monitoring time.

**A property of the proposed drift rule worth discussing.** After a retrain the drift reference is all labelled rows so far, which are still mostly pre-shift rows. A shifted window therefore keeps differing from the reference, and the drift-triggered strategy retrains on every window after a lasting shift. The same happens in the real replay, where the trigger fires against the training rows in every window (notebook 04). So in practice the drift-triggered strategy may retrain more often than the periodic one.

**Results (notebook 06, real data).** The full grid (576 window rows) ran in about 3 minutes. Every fit took under 1 second (largest 0.88 s); drift monitoring took about 0.2 s per window. Mean PR-AUC over the 8 windows, averaged over the 3 seeds:

| Scenario | Random guess | LR static | LR periodic | LR drift | SVM static | SVM periodic | SVM drift |
|---|---|---|---|---|---|---|---|
| Real replay | 0.308 | 0.399 | 0.400 | 0.400 | 0.361 | 0.400 | 0.398 |
| Stationary | 0.042 | 0.050 | 0.052 | 0.050 | 0.050 | 0.052 | 0.050 |
| Sudden shift | 0.044 | 0.063 | 0.068 | 0.069 | 0.062 | 0.067 | 0.068 |
| Gradual shift | 0.043 | 0.065 | 0.069 | 0.069 | 0.065 | 0.069 | 0.069 |

Retrains per run (same for both models and all seeds): static 0; periodic 3 in every scenario; drift-triggered 6 in the real replay, 0 in stationary, 4 in sudden and 4 in gradual.

- *Real replay.* The share of "yes" rises from 7% in window 1 to 54% in window 7, and every model's PR-AUC rises with it, staying above random guessing in every window. For Logistic Regression the three strategies are almost identical (0.399 to 0.400). For Linear SVM, retraining helps (static 0.361 vs. 0.400 periodic). The drift rule fires in all 8 windows, so the drift-triggered strategy retrained 6 times (the first two windows had no new labels yet) for no gain over periodic's 3 retrains.
- *Stationary.* The drift rule never fired (0 false alarms in 24 windows per model), so drift-triggered equals static exactly. Periodic retrained 3 times with a negligible difference (0.052 vs. 0.050).
- *Sudden and gradual shift.* No alerts before the change point. Detection delay: 0 windows for sudden and 2 windows for gradual, for all seeds. After the shift both retraining strategies beat static (about 0.068 vs. 0.063 for sudden), with drift-triggered and periodic close to each other.
- *Caution.* In the synthetic scenarios PR-AUC is only slightly above random guessing (about 0.05 to 0.07 vs. 0.04), because the pool comes from the early training rows (4.8% "yes") where the model has little signal. Differences of 0.005 should not be read as real without the test across seeds. The threshold-based metrics (F1, precision, recall) also move with the re-tuned cut-off: after retraining in the real replay, recall jumps to about 0.96 at the cost of precision.

## For Person 4 (results analysis)

```python
import pandas as pd
results = pd.read_csv('../results/results.csv')
# one row per (scenario, strategy, model, seed, window); window and change_window are 1-based
```

| Column | Meaning |
|---|---|
| `scenario`, `scenario_kind`, `strategy`, `model`, `feature_set`, `seed`, `window` | identify the row |
| `change_window`, `after_change` | first shifted window (1-based; empty for real replay and stationary); whether this window is at or after it |
| `pr_auc`, `random_pr_auc`, `roc_auc`, `f1`, `precision`, `recall`, `tp`, `fp`, `fn`, `tn`, `threshold` | performance on this window with the threshold used |
| `alert`, `alert_features`, `max_psi_trigger_groups` | drift rule fired on this window (against this strategy's current training rows) and which features |
| `economic_alert`, `max_psi_economic` | economic indicators, monitored only |
| `wanted_retrain`, `retrained`, `retrain_skipped_no_new_labels`, `retrain_seconds`, `threshold_source` | what the strategy did before scoring this window |
| `model_version`, `n_train_rows`, `labelled_windows_in_model` | which model scored this window |
| `initial_fit_seconds`, `initial_threshold`, `monitor_seconds` | cost of the shared initial fit and of monitoring |

Detection delay (scenarios sudden and gradual) = first window with `alert` at or after `change_window`, minus `change_window`. False alarms = alerts in the stationary scenario, plus alerts before `change_window` in sudden and gradual.
