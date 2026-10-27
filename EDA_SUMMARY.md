# EDA Summary (Step 1.7) — draft for team review

Source: Person 1's `notebooks/01_EDA.ipynb` (saved outputs). Every number below is traced to a section of that notebook in the table at the end.
No model has been trained yet. The 70/30 cut below is **for exploration only**; it is not our final train / development / stream split (still a pending decision).
Year-month labels reconstructed in the notebook are **not** used here; time is described by row position (UCI documents the file as ordered by date, May 2008 to November 2010, but it has no date column).

## Evidence levels (kept separate)

- **Observed (measured in notebook 01):** the subscription-rate and feature-distribution differences between early and late rows, reported below.
- **Possible (hypothesis, not measured):** a model fitted on earlier rows may perform worse on later rows because of these shifts.
- **Not yet measured:** any model's performance. No model has been trained, so there are no PR-AUC, ROC-AUC, F1, precision or recall results.

## Report paragraph

The UCI Bank Marketing file `bank-additional-full.csv` has 41,188 rows, 20 input features and a binary target `y` (term-deposit subscription). Only 11.3% of rows are "yes" (4,640 "yes" vs. 36,548 "no", about 7.9 : 1), so accuracy is misleading; PR-AUC is our primary metric and must be read against the random-guessing baseline, which equals the positive rate. There are no true missing values, but six categorical columns code missing information as "unknown" (20.9% of `default`), 12 rows are exact duplicates, and `pdays = 999`, a placeholder for "not previously contacted", appears in 96.3% of rows; `duration` is excluded because it is only known after a call. Comparing the first 70% of rows with the last 30% (an illustrative cut, not our modeling split), the subscription rate rises from 5.6% to 24.6%, and all five economic indicators shift strongly (KS statistics 0.59 to 1.00; PSI far above the conventional 0.25 "major shift" line), while the only two customer/campaign features tested, `age` and `campaign`, shift much less (KS 0.095 and 0.090; PSI 0.12 and 0.05). Because the positive rate differs between early and late rows, the random-guessing PR-AUC differs too, so PR-AUC values from different windows must be compared against their own baselines. These shifts motivate our study of static, periodic and drift-triggered retraining, since a model fitted on earlier rows may face different inputs and a different class balance later; they do not, however, show that any model's performance degrades or what causes the changes. Only our experiments can test that.

**Separate finding (Step 1.6, `notebooks/02_leakage_review.ipynb`, not part of Person 1's EDA):** beyond `duration`, whether `campaign`, `contact`, `month`, `day_of_week` and the five economic indicators were known before the call is unresolved, so no feature set is approved for modeling yet.

## Slide bullets

1. **Size and imbalance:** 41,188 rows, 20 inputs. Only 11.3% subscribe (about 7.9 : 1), so accuracy misleads. PR-AUC is primary, shown next to the random-guess baseline (the positive rate).
2. **Data quality:** no NaNs, but "unknown" is coded in 6 columns (`default` 20.9%). 12 duplicate rows. `pdays = 999` in 96.3% of rows. `duration` excluded (known only after the call).
3. **Earlier vs. later rows (illustrative 70/30 cut):** subscription rate 5.6% in the first 70% of rows, 24.6% in the last 30%.
4. **Shift evidence (KS / PSI):** all five economic indicators have KS 0.59 to 1.00 and PSI far above 0.25 (e.g. `euribor3m` mean 4.69 early vs. 1.14 late). `age` (KS 0.095, PSI 0.12) and `campaign` (KS 0.090, PSI 0.05) shift much less.
5. **Why it motivates our research:** the data's inputs and class balance change along the row order, so a model trained early may meet different data later. This is the situation retraining strategies are meant to handle.
6. **Caution:** these are exploratory differences. They do not prove performance degradation or a causal link. The PR-AUC baseline also moves with the positive rate.

## Method notes (kept short)

- **KS p-values** print as 0.0 for every tested feature. With about 41,000 rows even small differences give tiny p-values, so we report the KS statistic (size of the difference) instead.
- **PSI** here uses up to 10 quantile bins built from the early rows, with a small floor on bin proportions. For the economic indicators many bins are empty in one of the two periods, so their exact PSI values (4.48 to 11.52) depend on these choices and should not be over-interpreted. A separate sensitivity check, run outside notebook 01 and not part of its outputs, found that changing the floor changes these PSI values substantially while they stayed above 0.25 in every variant tried. The conclusion "far above 0.25" is therefore the safe statement; `age` and `campaign` have no empty bins and were not affected.
- Only 7 features were tested (`age`, `campaign` and the five economic indicators), so "customer and campaign features shift little" applies only to those two.

## Where each number comes from (`notebooks/01_EDA.ipynb`)

| Claim | Notebook section | Verified value |
|---|---|---|
| 41,188 rows, 21 columns (20 inputs + `y`) | 1. Load and confirm shape | `Shape: (41188, 21)` |
| 36,548 no / 4,640 yes, 88.7% / 11.3% | 3. Target class distribution (figure `01_target_distribution.png`) | counts and percentages in the chart; also confirmed independently from the raw file |
| About 7.9 : 1 | 3. Target class distribution | `Imbalance ratio (no : yes) = 7.9 : 1` |
| No true NaNs; 12 duplicates; `pdays = 999` 96.3% | 2. Data quality | `True NaN values: 0`, `Duplicate rows: 12`, `96.3%` |
| 6 columns with "unknown"; `default` 20.9% | 2. Data quality | `unknown_pct` table (`default` 20.87) |
| `duration` excluded | 6. Leakage check; 7. Save the shared dataset | `duration` removed in the saved file |
| Yes rate 5.6% early vs. 24.6% late | 4. Early vs. late window | `Yes rate — early: 5.6%  late: 24.6%` |
| KS 0.586 to 0.996 (economic); `age` 0.095; `campaign` 0.090 | 4. Early vs. late window (drift table) | `KS_stat` column |
| PSI 4.48 to 11.52 (economic); `age` 0.117; `campaign` 0.050 | 4. Early vs. late window (drift table) | `PSI` column |
| `euribor3m` mean 4.69 vs. 1.14 | 4. Early vs. late window (drift table) | `early_mean` 4.685, `late_mean` 1.138 |
