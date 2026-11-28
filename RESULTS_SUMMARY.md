# Results summary: what happened and why the three strategies performed alike

Plain-language findings for the report. **Every number is tagged with the section of `notebooks/07_results.ipynb` it comes from** (for example `[07 A3]`); the tables behind them are in `results/tables/07_*.csv` and the figures in `figures/14_*` to `figures/23_*`. Source data: `results/results_v2.csv` (primary; real-replay rows equal v1, synthetic rows are the corrected ones). `results/results.csv` (v1) appears only in section A6.

**Status labels.** VERIFIED = computed from saved results in this notebook. DIAGNOSTIC = uses extra information or a re-run and must not be used to choose settings. HYPOTHESIS = a plausible explanation the data are consistent with but do not prove. All stream settings (8 windows, label delay 1, periodic every 2 windows, drift rule PSI > 0.25, 3 seeds, expanding history, re-tuned cut-off) are **PROPOSED** defaults, not team-confirmed decisions (`DECISIONS.md`).

PR-AUC (precision-recall area under the curve) is the primary metric. The random line is the share of "yes" in a window. Lift = PR-AUC minus the random line.

## 1. The short answer

1. **VERIFIED. In the real data the three strategies are almost indistinguishable for Logistic Regression** (mean PR-AUC 0.3989 static, 0.4005 periodic, 0.4001 drift-triggered; random line 0.3083) `[07 A3, fig 14]`. The 95% bootstrap interval for periodic minus static is -0.007 to +0.011 and includes zero `[07 C]`.
2. **VERIFIED. Linear SVM is the exception: its PR-AUC was higher after retraining** (0.3606 static vs 0.3998 periodic and 0.3981 drift-triggered `[07 A3]`). The mean difference over the 8 windows is +0.0393 (95% interval +0.0266 to +0.0522) for periodic minus static and +0.0376 (+0.0249 to +0.0506) for drift-triggered minus static `[07 C]`. Read this narrowly: (a) the interval reflects within-window sampling variation only (which customers are in each window); (b) the real replay is a single deterministic run, so the three seeds are one result, not three; (c) the static Linear SVM started weaker than the static Logistic Regression (mean ROC-AUC 0.533 vs 0.590, mean PR-AUC 0.361 vs 0.399 `[07 A3]`) and the retrained SVM reaches only about the Logistic Regression level. **This does not establish a general retraining advantage.**
3. **VERIFIED. Drift-triggered retraining did not beat periodic retraining; it only retrained more often** (6 retrains vs 3 in the real replay) `[07 A3]`. The drift alarm fires in all 8 real windows, starting with window 1 `[07 A4]`, so it cannot tell windows apart.
4. **VERIFIED. In the corrected synthetic scenarios nothing separates the strategies.** Mean PR-AUC is 0.052 to 0.058 against a random line of 0.045 to 0.047, retraining gains over static are at most +0.0023 (over the windows after the first retrain), and every bootstrap interval includes zero `[07 A3, A4, C]`.
5. **We cannot say that drift-triggered retraining is better.** The evidence supports "no detectable difference" for Logistic Regression and for the synthetic scenarios, and "retraining helped a weaker model" for Linear SVM in one real stream.

## 2. Why the strategies performed alike (evidence for each reason)

**Reason 1. The models carry little signal, and raw PR-AUC has a moving floor. VERIFIED.**
- In the real replay the share of "yes" rises from 7.3% (window 1) to 54.2% (window 7) and PR-AUC rises from 0.120 to 0.671. About 85% of the rise in raw PR-AUC (0.469 of 0.551) matches the rise in the random-ranking baseline, which for PR-AUC equals the share of "yes" by definition. The model's lift above that baseline changed much less (0.047 in window 1 to 0.130 in window 7), so most of the raw increase reflects the metric's moving floor rather than better ranking. This is a property of the metric, not a statement about customers or the economy `[07 A5, fig 19]`.
- The correlation between the share of "yes" and PR-AUC is 0.995 across the 8 windows (descriptive, 8 points) `[07 A5]`.
- Static Logistic Regression's ROC-AUC is below 0.5 in window 3 (0.479) `[07 A5]`. Window 3 has a lift of 0.004.

**Reason 2. A retrain on an expanding history changes the model only a little. VERIFIED (numbers), HYPOTHESIS (that this is the cause).**
- A retrain refits on all labelled rows. After six retrains the drift-triggered model's training data is still only 20.0% stream rows (6,178 of 30,890); the periodic model's is 17.2% (5,148 of 29,860) `[07 A7, fig 23]`.
- The original rows have 4.8% "yes"; the stream windows have 7% to 54% `[07 A7]`. The rows that dominate training describe a very different world, yet the retrained model hardly moves.

**Reason 3. A model trained only on a window's own labels does not do much better than the existing models. DIAGNOSTIC, small-sample.**
- Within-window cross-validation diagnostic (5-fold cross-validation inside each window, **about 800 training rows per fold, against about 24,700 older rows for the static model**; it uses labels a deployment would not have; earlier drafts called it an "oracle ceiling", which it is not): mean PR-AUC 0.407 for Logistic Regression against 0.417 for the static model, 0.418 periodic, 0.419 drift-triggered, on the same held-out rows. Linear SVM: within-window 0.405, static 0.377, periodic 0.417 `[07 B, fig 20]`.
- For Logistic Regression the within-window model is below the static model in 4 of 8 windows (1, 2, 4 and 6) and above it in the other 4 by 0.003 to 0.044. In windows 1 and 2 it is far below (0.105 and 0.132 vs 0.165 and 0.161), because a window holds only about 1,000 rows. For Linear SVM it is above static in windows 5 to 8 (by 0.053 to 0.101), which is exactly where retraining recovers PR-AUC `[07 B]`.
- For Logistic Regression this gives no sign that a lack of recent labels is what limits the model at this sample size. It is a small-sample cross-validation diagnostic, **not a theoretical performance ceiling or an upper bound**: with about 30 times fewer training rows than the static model it cannot rule out gains from more recent data.

**Reason 4. The drift alarm carries no information in the real replay. VERIFIED.**
- It fires in 8 of 8 windows (the 10 model inputs are measured against the original training rows, which hold almost no repeat customers) `[07 A4, fig 18]`. Drift-triggered retraining therefore equals "retrain whenever new labels have arrived", which is the periodic schedule at double the frequency.
- In the synthetic scenarios the alarm works as designed: no false alarms (0 of 24 stationary windows, none before a change point), detection delay 0 windows (sudden) and 2 windows (gradual), for all seeds `[07 A4]`.

**Reason 5. Detected shifts do not hurt the model, so there is nothing to recover. VERIFIED for these scenarios.**
- The synthetic age shift is detected clearly (age PSI about 0.8 to 1.1 after the change) but the static model's lift does not drop: mean change after the change point is -0.0021 (sudden) and +0.0010 (gradual); the per-seed values are mixed (sudden: -0.0070, +0.0067, -0.0059) `[07 A4]`.
- Models in the synthetic scenarios are close to random anyway (lift at most 0.010) `[07 A3]`.

## 3. TA question 2: do alerts line up with real performance drops, and did retraining after an alert help?

- **Alerts vs performance drops. VERIFIED, no evidence of a match.**
  - *Real replay:* the alert is on in every window, so it cannot separate good from bad windows. The alarm size (largest PSI) is positively correlated with static lift (Spearman 0.90) and PR-AUC (0.76), but both simply rise over time as the stream moves away from the training period; this is a trend, not a drop `[07 A4]` (8 points, descriptive).
  - *Synthetic:* alerts are correct about the built-in change but the static model does not get worse after it (Reason 5).
- **Retraining after an alert. VERIFIED.** Mean PR-AUC gain over static in the 6 windows scored after the first retrain: real Logistic Regression +0.0017 (drift-triggered) and +0.0021 (periodic), better than static in 4 of 6 windows; real Linear SVM +0.050 and +0.052, better in all 6; synthetic scenarios between -0.0003 and +0.0023, better than static in 1.7 to 4 of the 4 to 6 windows `[07 A4]`.
- **Real Linear SVM is one stream.** The three seeds are identical for the real replay, so "3 of 3 seeds exclude zero" is one result, not three `[07 A1]`.

## 4. Cost

- Retraining takes under 1 s per fit. Per real-replay run: Logistic Regression 1.82 s of retraining for periodic (3 retrains) vs 3.47 s for drift-triggered (6 retrains); Linear SVM 0.72 s vs 1.48 s `[07 A3]`. These are the timings measured in the `results_v2.csv` run on this machine.
- Drift monitoring costs about 1.3 to 1.4 s per real-replay run. It is logged for every strategy, but only the drift-triggered strategy needs it, so drift-triggered pays both costs `[07 A3]`.
- So drift-triggered costs about twice the retraining time of periodic plus the monitoring time, for no additional PR-AUC in the real replay. In absolute terms every cost is small.

## 5. Threshold-based metrics (read with care)

F1, precision and recall use each window's recorded cut-off: the best-F1 cut-off on the most recent 1,030 labelled rows, re-tuned at every fit. After retraining, Logistic Regression's recall rises from 0.511 (static) to 0.639 (periodic) and 0.624 (drift-triggered), precision falls from 0.359 to 0.323 and 0.326, and F1 barely moves (0.406, 0.413, 0.408) `[07 A3]`. That pattern is consistent with the re-tuned cut-off trading precision for recall more than with a better ranking (PR-AUC is almost unchanged). The static model keeps one cut-off (0.5228) for all windows.

## 6. Uncertainty (diagnostic bootstrap)

- Method: 2,000 resamples of the rows within each window, the same resample for all strategies, 95% percentile intervals of the mean PR-AUC difference over the 8 windows `[07 C]`. The fast PR-AUC used was validated against scikit-learn (largest difference 1.1e-16, limit 0.001) and the per-row scores reproduce `results_v2.csv` exactly (8,064 values compared, 0 differ) `[07 per-row scores]`.
- Real replay, Logistic Regression: periodic minus static +0.0016 (95% interval -0.0066 to +0.0108), drift-triggered minus static +0.0013 (-0.0073 to +0.0102). Real replay, Linear SVM: +0.0393 (+0.0266 to +0.0522) and +0.0376 (+0.0249 to +0.0506) `[07 C, fig 21]`. These intervals reflect within-window sampling variation only, and the real replay is one deterministic run (see finding 2 in section 1 for how to read the Linear SVM result).
- Synthetic scenarios: every interval includes zero; interval ends between about -0.007 and +0.009. In the stationary scenario drift-triggered never retrains, so it equals static exactly (interval 0 to 0) `[07 C]`.

## 7. Error analysis (descriptive only)

Real replay, static Logistic Regression at its fixed cut-off 0.5228, all 8 windows pooled (windows 4 to 8 hold most subscribers): of 2,540 subscribers it catches 1,493 (recall 0.588) and misses 1,047, while predicting "yes" for 3,669 of 8,238 rows `[07 D, fig 22]`.
- **Previous campaign outcome dominates.** It catches 99.5% of subscribers whose previous outcome was a success (827 of 831) but only 22.3% after a previous failure (107 of 480) and 45.5% with no previous contact (559 of 1,229). Of the 1,047 missed subscribers, 1,043 had no successful previous campaign `[07 D]`.
- Recall is 78% for customers under 30, about 42% for ages 40 to 59 and about 59% for the 30 to 39 and 60+ groups; students (99.5%, 214 of 215) and retired customers (69%) are caught most, unemployed (39%) and housemaid (39%, 67 subscribers) least `[07 D]`.
- This describes the model; it does not explain why these customers are missed.

## 8. v1 versus v2 (synthetic scenarios only)

The apparent retraining gain over static shrinks from v1 to v2: Logistic Regression periodic minus static, sudden +0.0054 to +0.0003, gradual +0.0040 to +0.0017, stationary +0.0020 to +0.0010; drift-triggered minus static, sudden +0.0062 to +0.0006, gradual +0.0042 to +0.0008 `[07 A6]`. This is consistent with row reuse having inflated v1, but **v1 and v2 differ in more than one factor** (v2 also fits the initial model on 60% instead of 80% of the training rows and uses different random draws), so the comparison does not isolate the row reuse as the only cause. The real-replay rows are identical in both files (notebook 06b).

## 9. Limitations (state these in the report)

1. **One real stream.** The real replay is the only real-data evidence, 8 windows from one period, and it is deterministic: the 3 seeds are identical, so they are not independent replications.
2. **The synthetic seeds are not independent either.** They resample one fixed set of training rows. Synthetic scenarios are controlled experiments with separate scoring and history pools, not chronological replays, and they assume the arriving labels are a fresh sample of the window's distribution, not the same customers.
3. **The bootstrap intervals** reflect which customers appear in each window, not training randomness (the models are deterministic), and say nothing about other periods or other data. No significance test across seeds is claimed.
4. **The within-window cross-validation is a small-sample diagnostic, not a performance ceiling** (about 800 training rows per fold from windows of about 1,000 rows, 5 folds, against about 24,700 rows for the static model; it uses labels a deployment would not have). It must not be used to choose any setting. Synthetic windows were skipped because duplicate rows would leak across folds.
5. **The models are weak** (real-replay lift about 0.09; ROC-AUC 0.48 to 0.67 per window; synthetic lift at most 0.010). Differences between strategies are small partly because there is little signal to recover.
6. **The drift rule is not selective in the real replay** (fires in all windows), so the real replay cannot test whether a *selective* trigger would help. Sliding windows and other drift rules are future work and were not tried.
7. **All settings are PROPOSED defaults**, chosen before these results; none was tuned on them. The stream results for those defaults have now been seen by the team, so any later change must be reported as a secondary analysis.
8. **F1, precision and recall depend on re-tuned cut-offs** and should not be read as model-quality differences alone.
9. **Error analysis pools windows** whose share of "yes" differs strongly (7% to 54%).
10. **Timing values** are single measurements on one machine; v1 timings (another machine) were not combined with v2 timings.
