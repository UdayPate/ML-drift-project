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

# pdays = 999 is a sentinel, not a number of days. The UCI documentation says it means "not previously
# contacted", but 4,110 rows have previous > 0 together with pdays = 999, so we do not assume it always
# means that. Those rows are kept. (Of them, 124 fall in the training rows and 1,977 in the development
# rows, so the pattern is mostly in later data.)
PDAYS_SENTINEL = 999
