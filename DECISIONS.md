# DECISIONS.md — Project Decision Log

Format: `YYYY-MM-DD | Decision | Why`

2026-10-26 | Use the UCI Bank Marketing `bank-additional-full.csv` dataset | It offers chronological observations and interpretable customer, campaign, and economic features for classical-ML drift experiments.
2026-10-26 | Preserve dataset row order for real-data analysis | The published file is ordered chronologically but lacks exact per-record dates.
2026-10-26 | Exclude call duration for pre-call predictions | Duration is not available at prediction time and would introduce leakage.
2026-10-26 | Initially compare Logistic Regression, Decision Tree, Random Forest, and SVM | These classical models match the course focus and offer different modeling approaches.
2026-10-26 | Shortlist roughly 2–3 models before extensive sequential experiments | This retains a broad initial comparison while controlling experiment cost.
2026-10-26 | Compare static, periodic, and drift-triggered retraining | This directly tests the central research question.
2026-10-26 | Keep four core scenarios: real chronological, stationary baseline, sudden shift, and gradual shift | Together they provide real-data evidence and controlled tests with manageable scope.
2026-10-26 | Treat concept drift, SHAP, and Streamlit as optional | Prioritize rigorous classical-ML results before extensions.
2026-10-26 | Exclude deep learning, LLM reports, and production deployment infrastructure | Stay aligned with course expectations and timeline.
2026-10-26 | Use a tutor-first, one-step-at-a-time coding-agent workflow | Each implementation step should be explained, tested, and understood before proceeding.
2026-10-26 | Human student alone controls Git commits | The agent must never stage by default or commit, push, or rewrite history.
2026-10-26 | Keep the GitHub repository public | Team choice; course work is shared openly.
2026-10-27 | Use Moment B: predict subscription immediately before a planned marketing call begins | Defines the prediction time so only information available before the call can be used. Current call duration and outcome are unavailable. This does not automatically approve ambiguous features.
2026-10-27 | Add scikit-learn to `requirements.txt` (pinned 1.9.1) | Needed for the Pipeline/ColumnTransformer preprocessing and the Logistic Regression baseline; installed in the existing venv and checked with `pip check`.
2026-10-27 | Notebooks import `src/` through a simple project-root `sys.path` line; no packaging | Simplest approach that works for today's milestone and adds no extra configuration files.
2026-10-27 | Use standard-library `unittest`; do not add pytest for this milestone | The Phase A checks are small; avoids a new dependency.
2026-10-27 | Keep categorical `unknown` as its own category | It is a real answer value in the data, not a missing entry; it avoids dropping or guessing values.
2026-10-27 | Treat `pdays = 999` as a sentinel: add indicator `pdays_is_999` and keep the real `pdays` value (median-imputed, median learned from training rows only) for all other rows; keep all rows | 999 is not a number of days. 4,110 rows have `previous > 0` with `pdays = 999`, so we do not claim 999 always means "never contacted"; those rows are kept and documented, not deleted.
2026-10-27 | Initial Logistic Regression baseline uses 10 conservative features: `age`, `job`, `marital`, `education`, `default`, `housing`, `loan`, `pdays`, `previous`, `poutcome` | Provisionally available before the call under Moment B. Excludes `duration`, current-contact features (`campaign`, `contact`, `month`, `day_of_week`) and the economic indicators, whose availability is unresolved. This does not decide whether those features are used later.
2026-10-27 | Use a 60/20/20 chronological split of the full 41,188 rows for the initial baseline: train = rows 0–24,711 (24,712), development = rows 24,712–32,949 (8,238), stream = rows 32,950–41,187 (8,238); 0-based row index, original order, no shuffling | Methodological goal: train on the past, evaluate on the next period, keep a future stream untouched. Feasibility inspection: train 1,188 yes of 24,712 (4.81%), dev 912 yes of 8,238 (11.07%), so both have enough positives. The stream must not influence preprocessing, fitting, tuning or split selection. The split is not to be changed later to improve model scores.
2026-10-27 | Do not standardize `pdays_days` or `previous`; keep median imputation of `pdays_days` (median from training rows only) and the separate `pdays_is_999` indicator; still scale `age` and one-hot encode categoricals | In the training rows only 6 of 24,712 have a real `pdays` value and `previous` never exceeds 1, so a scaler would get a tiny spread (about 0.016 for `pdays_days`) and later rows would scale to extreme values (up to about ±513). `pdays_days` rests on only 6 real observations, so any relationship learned for it may be unreliable.

<!-- Add confirmed decisions below, one per line. Do not add proposals as decisions. -->


## Pending decisions
Open questions. Do not treat any of these as decided. When one is resolved, delete it here and add a dated line to the log above.

- Monitoring window size (rows per window)
- Include economic indicators as model features, exclude them, or test both
- Primary decision threshold: fixed 0.5, or re-tuned on recent labelled data at each retrain
- Retraining data: expanding window (all available past rows) or sliding window (most recent N rows)
- Periodic retraining interval (every how many windows)
- Drift-trigger rule: which test(s), which feature group(s), which threshold
- Label delay: how many windows before true labels count as available
- Which 2–3 models to shortlist after the initial comparison
- Number of random seeds and the statistical test used to compare strategies
- Whether reconstructed year-month labels may be used for EDA/labelling (windows stay row-index based)
- Whether campaign, contact, month and day_of_week were known before the modeled call (is a row one call or a whole campaign? planned vs. actual values)
- When each economic indicator's value was publicly available relative to call dates (documentation gives measurement frequency only, no publication dates or delays)
