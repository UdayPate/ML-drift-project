# PROGRESS.md — Step Checklist

Work through one unticked step at a time, in order. A step is ticked only after the student has reviewed it and says it is done.
Each step should be small enough to explain, check, and understand in one sitting.

Legend: `[ ]` not started · `[~]` in progress · `[x]` done (student-approved)

---

## Phase 0 — Project setup
- [x] 0.1 Write the agent rules and logs (`AGENTS.md`, `CLAUDE.md`, `PROGRESS.md`, `DECISIONS.md`, `LEARNING_LOG.md`)
- [ ] 0.2 Create the folder layout, `requirements.txt`, and `.gitignore` (student creates the Git repository and makes the first commit)
- [ ] 0.3 Set up a Python virtual environment and confirm the packages install and import

## Phase 1 — Data understanding and EDA (Exploratory Data Analysis)
- [ ] 1.1 Download `bank-additional-full.csv` from UCI; load it and verify row count, columns, and target counts against the UCI documentation
- [ ] 1.2 Inspect data types, `unknown` values, `pdays = 999`, duplicates, and near-constant columns
- [ ] 1.3 Class imbalance: overall positive rate
- [ ] 1.4 Timeline: how rows and the positive rate are spread across the row order (see "Checks to run" in `AGENTS.md`)
- [ ] 1.5 How customer, campaign, and economic features change across the row order
- [ ] 1.6 Leakage review: confirm `duration` is excluded; check every other feature for availability before the call
- [ ] 1.7 Write a one-paragraph EDA summary for the report

## Phase 2 — Shared setup in `src/`
- [ ] 2.1 Settings module: feature groups, seed, paths (split and window values stay pending until decided)
- [ ] 2.2 Data loading and cleaning functions (decide `pdays` / `unknown` handling first)
- [ ] 2.3 Split check: compare candidate train/dev/stream splits with a simple baseline → decide the split
- [ ] 2.4 Window check: compare candidate window sizes → decide the window size
- [ ] 2.5 Preprocessing pipeline (scaling + one-hot encoding), fitted on training rows only
- [ ] 2.6 Tests: row order never shuffled, `duration` never used, preprocessing never fitted on stream rows

## Phase 3 — Model comparison (Stage 1) and shortlist (Stage 2)
- [ ] 3.1 Logistic Regression baseline on train → dev
- [ ] 3.2 Decision Tree
- [ ] 3.3 Random Forest
- [ ] 3.4 SVM (Support Vector Machine)
- [ ] 3.5 Hyperparameter tuning with time-ordered validation
- [ ] 3.6 With vs. without economic features → decide
- [ ] 3.7 Compare all models on dev (PR-AUC primary) → shortlist 2–3 models

## Phase 4 — Drift detection
- [ ] 4.1 KS (Kolmogorov–Smirnov) test for numeric features, per window
- [ ] 4.2 PSI (Population Stability Index), per window
- [ ] 4.3 Run both separately on customer/campaign features vs. economic features
- [ ] 4.4 Calibrate drift thresholds on development data only → decide the drift-trigger rule
- [ ] 4.5 Do alerts line up with real performance drops? (false-alarm analysis)

## Phase 5 — Scenarios
- [ ] 5.1 Scenario 0: real chronological replay (stream setup, label delay decided)
- [ ] 5.2 Scenario 1: stationary baseline (fixed reference distribution, no leakage)
- [ ] 5.3 Scenario 2: sudden covariate drift via documented resampling/reweighting
- [ ] 5.4 Scenario 3: gradual covariate drift via documented resampling/reweighting
- [ ] 5.5 Validate each scenario: confirm the intended shift actually appears in the data

## Phase 6 — Retraining strategies (main experiment)
- [ ] 6.1 Strategy A: static model
- [ ] 6.2 Strategy B: periodic retraining (decide interval and expanding vs. sliding window)
- [ ] 6.3 Strategy C: drift-triggered retraining
- [ ] 6.4 Run all strategies × scenarios × shortlisted models × seeds; save raw results to a file
- [ ] 6.5 Record compute time and retraining counts

## Phase 7 — Evaluation and analysis
- [ ] 7.1 Performance over time plots (with the random-guessing PR-AUC line)
- [ ] 7.2 Strategy comparison tables: PR-AUC, ROC-AUC, F1, retrains, detection delay, false alarms, time
- [ ] 7.3 Statistical test across seeds → are the differences real?
- [ ] 7.4 Feature importance (coefficients, permutation importance) before vs. after shifts
- [ ] 7.5 Write up findings, including cases where retraining did not help

## Phase 8 — Optional extensions (only after Phase 7)
- [ ] 8.1 Concept drift scenario
- [ ] 8.2 SHAP analysis
- [ ] 8.3 Lightweight Streamlit results viewer for the demo video

## Phase 9 — Deliverables
- [ ] 9.1 Report (journal template, 10–12 pages, two columns) and Turnitin check (< 25% overlap)
- [ ] 9.2 Final slides
- [ ] 9.3 Demo video (12–15 minutes)
- [ ] 9.4 Clean repository: README, reproducible run instructions, all notebooks run top to bottom
