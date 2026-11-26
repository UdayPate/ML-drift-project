# Drift detection and scenarios: method note and handoff

Person 1's part of the final project. Code: `src/drift.py`, `src/scenarios.py`. Evidence: `notebooks/04_drift_and_scenarios.ipynb`. Tests: `tests/test_drift.py`, `tests/test_scenarios.py`. All settings are in `src/settings.py` and are **PROPOSED, pending team approval** (see `DECISIONS.md`).

## For the report (method note)

**Drift detection.** For each stream window we compare the window with a reference sample, normally the rows the current model was trained on. Two measures are computed per feature. The Population Stability Index (PSI) compares the share of rows per bin: numeric features use 10 quantile bins taken from the reference only, while text features and numbers with at most 10 distinct values use one bin per value; empty bins get a floor of 1e-4. The Kolmogorov-Smirnov (KS) statistic, the largest gap between the two cumulative distributions, is reported for numeric features as an effect size; its p-value is not used, because with thousands of rows almost any difference is "significant". Features are monitored in two groups: the 10 model inputs (customer and contact-history features) and the 5 economic indicators. The proposed trigger rule retrains the model when any model-input feature has PSI above 0.25, the common "major shift" rule of thumb. Economic indicators are measured and logged but do not trigger retraining while they are not model inputs.

**Checking the threshold without the stream.** To see how the rule behaves when nothing changes, we drew 200 random samples of 1,030 training rows and compared each with all training rows. The largest PSI of any feature was 0.026, and the rule never fired (estimated false-alarm rate 0.0 per window). The stream was not used to choose the threshold.

**Scenarios.** Every retraining strategy is run on the same four streams of 8 windows:

| Scenario | Kind | How it is built | Purpose |
|---|---|---|---|
| Real replay | Real | The last 20% of rows (8,238) in original order, cut into 8 windows of ~1,030 rows; initial model fitted on the first 60% | Behaviour under real change; no known change point |
| Stationary | Synthetic | Every window drawn at random from one fixed pool | Counts false alarms and needless retrains |
| Sudden shift | Synthetic | As stationary, then from window 5 rows are drawn with weights favouring older customers | Detection delay with a known change point |
| Gradual shift | Synthetic | The same shift growing from window 3 to full strength at window 7 | Detection delay under slow change |

For the synthetic scenarios a fixed random 20% of the training rows is held out as the sampling pool and the other 80% fit the initial model, so evaluation never reuses training rows. Windows are random samples of whole observed rows (features and label together), so no values are invented. The shift draws rows with probability proportional to exp(1.0 × standardised age). These scenarios are controlled experiments, not historical evidence.

**Results of the checks (notebook 04, seeds 42, 43, 44).**

| Scenario | Alerts | Detection delay |
|---|---|---|
| Stationary | 0 of 24 windows | not applicable |
| Sudden | none before the change; every shifted window | 0 windows (all 3 seeds) |
| Gradual | none before the change; from window 5 on | 2 windows after the shift starts (all 3 seeds) |
| Real replay | 8 of 8 windows against the training rows | no known change point |

In the real stream the model inputs that move most are `previous` and `poutcome` (many more repeat customers later; PSI 1.1 to 4.0), followed by `pdays`, `job`, `default` and `age`. The economic indicators shift far more (PSI about 9 to 18, inflated by the empty-bin floor because their few values barely overlap between periods). In the synthetic scenarios the age shift is clearly detected (PSI about 1.0) while the share of "yes" stays about 4 to 5%, so they directly test whether input drift actually hurts predictions.

**Limits.** PSI depends on binning and the floor, and its 0.25 cut-off is a rule of thumb. All thresholds and window settings are proposals. An alert says the inputs changed, not that predictions got worse; that is measured in the strategy experiment.

Figures: `figures/10_null_psi_calibration.png`, `figures/11_scenario_drift_alarm.png`, `figures/12_real_replay_drift_by_group.png`.

## For Person 3 (strategy loop)

```python
import sys; sys.path.insert(0, '..')
from src.data import load_dataset
from src.scenarios import all_scenarios
from src.drift import drift_trigger
from src.settings import SEEDS, LABEL_DELAY_WINDOWS, PERIODIC_EVERY_WINDOWS

df = load_dataset()
for seed in SEEDS:
    for scenario in all_scenarios(df, seed=seed):     # real_replay, stationary, sudden, gradual
        reference = scenario.initial_train            # rows the current model was fitted on
        for i, window in enumerate(scenario.windows):  # i is 0-based; scenario.change_window is 0-based too
            decision = drift_trigger(reference, window)
            # decision['retrain'] -> True/False, decision['fired'] -> features above 0.25,
            # decision['report'] -> every monitored feature (log the economic group separately)
            # When the model is retrained, set `reference` to the rows it was retrained on.
```

Notes:
- Each `Scenario` has `name`, `kind` ('real'/'synthetic'), `initial_train`, `windows`, `change_window` (0-based, `None` when unknown), `description`, `strengths`.
- Windows keep every column of the original file, including `duration`; select model features with the existing `src/features.py` functions as usual.
- Retraining may use only earlier windows whose labels are available under `LABEL_DELAY_WINDOWS`, never the current or a future window (AGENTS.md).
- `drift_trigger` takes about 0.2 s per window (measured), so 4 scenarios × 8 windows × 3 seeds adds under 30 seconds per model.
