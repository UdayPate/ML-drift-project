"""Shared project settings. Import these everywhere; never copy them into notebooks.

Notebooks import this package with a project-root `sys.path` line, for example
(from a notebook inside `notebooks/`):

    import sys; sys.path.insert(0, '..')
    from src.settings import SEED
"""
from pathlib import Path

SEED = 42

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = PROJECT_ROOT / 'data' / 'raw' / 'bank-additional-full.csv'

# The full UCI file (not the 4,119-row sample).
EXPECTED_ROWS = 41_188
EXPECTED_COLUMNS = 21          # 20 inputs + the target

TARGET = 'y'
POSITIVE_LABEL = 'yes'

# Chronological split of the original row order (no shuffling). 60/20/20 was confirmed on
# 2026-10-27 for the initial baseline. Do not change it to improve model scores.
TRAIN_FRAC = 0.6
DEV_FRAC = 0.2
STREAM_FRAC = 0.2              # future stream: must not influence preprocessing, fitting or tuning

# Initial decision threshold for threshold-based metrics (precision, recall, F1, confusion matrix).
# It is a documented starting point only, not tuned; the project-wide choice is still a pending decision.
# A row is predicted "yes" when its predicted probability is >= this value.
DECISION_THRESHOLD = 0.5

# Initial Logistic Regression baseline: library defaults except for enough iterations to converge.
# No class weighting and no tuning.
LOGREG_MAX_ITER = 1000

# --- Feature sets (see notebooks/02_leakage_review.ipynb and DECISIONS.md) ---
ALWAYS_EXCLUDED = ['duration']   # only known after the call ends

# Initial baseline: 10 features provisionally available before the planned call (Moment B).
CUSTOMER_FEATURES = ['age', 'job', 'marital', 'education', 'default', 'housing', 'loan']
HISTORY_FEATURES = ['pdays', 'previous', 'poutcome']
BASELINE_FEATURES = CUSTOMER_FEATURES + HISTORY_FEATURES

# Not used in the first baseline: availability before the call is unresolved.
CURRENT_CONTACT_FEATURES = ['campaign', 'contact', 'month', 'day_of_week']
ECONOMIC_FEATURES = ['emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed']

# ------------------------------------------------------------------------------------------------------
# Stream, drift-monitoring and scenario settings.
# PROPOSED on 2026-11-25 as starting defaults so the team can build in parallel. They are NOT confirmed
# decisions yet (see "Pending decisions" in DECISIONS.md). If the team changes one, change it here only;
# every module and notebook reads these values from this file.
# ------------------------------------------------------------------------------------------------------

# Random seeds for repeated runs (PROPOSED: 3 seeds). SEED above stays the default for single runs.
SEEDS = [42, 43, 44]

# The stream (last 20% = 8,238 rows) is cut into this many consecutive windows of ~1,030 rows each,
# by row index only (no invented dates). PROPOSED.
N_STREAM_WINDOWS = 8

# How many windows pass before a window's true labels count as available for retraining. PROPOSED.
LABEL_DELAY_WINDOWS = 1

# Periodic strategy: retrain every this many windows. PROPOSED.
PERIODIC_EVERY_WINDOWS = 2

# --- Drift monitoring (src/drift.py) ---
# Feature groups are monitored separately (AGENTS.md: economic indicators apart from customer attributes).
MONITOR_GROUPS = {
    'customer_history': CUSTOMER_FEATURES + HISTORY_FEATURES,   # the 10 baseline model inputs
    'economic': ECONOMIC_FEATURES,
}
# Groups whose alerts can trigger a retrain. PROPOSED: only the features the model actually uses.
# If the team decides to add economic indicators as model features, add 'economic' here too.
TRIGGER_GROUPS = ['customer_history']

PSI_BINS = 10            # quantile bins for numeric features with more than PSI_BINS distinct values
PSI_EPS = 1e-4           # floor for empty bins, the same floor used in notebooks/01_EDA.ipynb
PSI_ALERT = 0.25         # PROPOSED trigger: retrain when any trigger-group feature has PSI above this
                         # (the common "major shift" rule of thumb, also used in the EDA)

# --- Synthetic scenarios (src/scenarios.py) ---
# Scenarios 2-4 sample observed (features, label) rows from a held-out part of the TRAINING rows, so their
# stream starts from the same distribution the initial model was trained on and never reuses its rows.
SYNTHETIC_POOL_FRAC = 0.2           # share of training rows held out as the sampling pool (PROPOSED)
SYNTHETIC_WINDOW_ROWS = 1_030       # same size as a real stream window
SHIFT_FEATURE = 'age'               # feature whose distribution is shifted (a model input) (PROPOSED)
SHIFT_STRENGTH = 1.0                # tilt per standard deviation of SHIFT_FEATURE (PROPOSED)
SUDDEN_CHANGE_WINDOW = 4            # 0-based: windows 4..7 are shifted (the 5th window onwards)
GRADUAL_START_WINDOW = 2            # 0-based: the shift grows from window 2 ...
GRADUAL_END_WINDOW = 6              # ... and reaches full strength at window 6

# pdays = 999 is a sentinel, not a number of days. The UCI documentation says it means "not previously
# contacted", but 4,110 rows have previous > 0 together with pdays = 999, so we do not assume it always
# means that. Those rows are kept. (Of them, 124 fall in the training rows and 1,977 in the development
# rows, so the pattern is mostly in later data.)
PDAYS_SENTINEL = 999

# ------------------------------------------------------------------------------------------------------
# Retraining-strategy experiment (src/strategies.py, Person 3).
# PROPOSED on 2026-11-26, NOT confirmed decisions (see "Pending decisions" in DECISIONS.md).
# ------------------------------------------------------------------------------------------------------

# The three strategies, in the order used in tables and plots.
STRATEGIES = ['static', 'periodic', 'drift_triggered']

# Models used in the stream experiment: Person 2's provisional shortlist (notebooks/05_models.ipynb), each with
# the index of its selected configuration in src.model_comparison.model_candidates(). Hyperparameters stay
# fixed during the stream; only the model weights and the decision threshold are refitted at a retrain.
EXPERIMENT_MODELS = {'Logistic Regression': 1, 'Linear SVM': 1}   # both class_weight='balanced' (verified 2026-11-26 by re-running Person 2's selection)
EXPERIMENT_FEATURE_SET = 'baseline_10'     # Person 2: economic indicators made every model worse on dev

# Retraining data: 'expanding' = every labelled row available so far (initial training rows plus the stream
# windows whose labels have arrived). PROPOSED default from the work plan.
RETRAIN_DATA = 'expanding'

# Decision threshold rule, used for the initial fit AND at every retrain, in every scenario and strategy:
# hold out the most recent THRESHOLD_TUNE_ROWS labelled rows, fit on the rest, pick the cut-off with the best
# F1 on the held-out rows, then refit on all labelled rows and keep that cut-off. PROPOSED.
THRESHOLD_TUNE_ROWS = 1_030              # one window's worth of rows
THRESHOLD_MIN_POSITIVES = 10             # fewer "yes" rows than this in the held-out rows: keep the old cut-off
