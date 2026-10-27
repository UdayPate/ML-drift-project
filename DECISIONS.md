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

<!-- Add confirmed decisions below, one per line. Do not add proposals as decisions. -->


## Pending decisions
Open questions. Do not treat any of these as decided. When one is resolved, delete it here and add a dated line to the log above.

- Train / development / stream split fractions (run the split check in `AGENTS.md` first)
- Monitoring window size (rows per window)
- Include economic indicators as model features, exclude them, or test both
- Primary decision threshold: fixed 0.5, or re-tuned on recent labelled data at each retrain
- Retraining data: expanding window (all available past rows) or sliding window (most recent N rows)
- Periodic retraining interval (every how many windows)
- Drift-trigger rule: which test(s), which feature group(s), which threshold
- Label delay: how many windows before true labels count as available
- How to handle `pdays = 999` ("never contacted before") and the `unknown` category values
- Which 2–3 models to shortlist after the initial comparison
- Number of random seeds and the statistical test used to compare strategies
- Whether reconstructed year-month labels may be used for EDA/labelling (windows stay row-index based)
