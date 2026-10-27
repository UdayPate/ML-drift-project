"""Feature selection and leakage-safe preprocessing for the initial baseline.

The preprocessing object learns its statistics (medians, means, category lists) only when `.fit()` is called.
Call `.fit()` on training rows only; use `.transform()` for development and stream rows.
"""
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from src.settings import (ALWAYS_EXCLUDED, BASELINE_FEATURES, CURRENT_CONTACT_FEATURES, CUSTOMER_FEATURES,
                          ECONOMIC_FEATURES, PDAYS_SENTINEL, TARGET)

SCALED_COLUMNS = ['age']                        # standardised (mean 0, spread 1), fitted on training rows
IMPUTED_COLUMNS = ['pdays_days']                # median-imputed (median from training rows), NOT standardised
PASSTHROUGH_COLUMNS = ['previous', 'pdays_is_999']   # used as they are
CATEGORICAL_COLUMNS = [c for c in CUSTOMER_FEATURES if c != 'age'] + ['poutcome']


def select_baseline_features(df):
    """Return only the 10 baseline input columns, refusing anything that must not be used."""
    banned = set(ALWAYS_EXCLUDED + CURRENT_CONTACT_FEATURES + ECONOMIC_FEATURES + [TARGET])
    overlap = banned & set(BASELINE_FEATURES)
    if overlap:
        raise ValueError(f'Baseline features must not include {sorted(overlap)}')
    return df[BASELINE_FEATURES].copy()


def add_pdays_features(X):
    """Replace `pdays` by two columns, because 999 is a sentinel and not a number of days.

    - `pdays_is_999`: 1 where pdays == 999, else 0.
    - `pdays_days`: the real value of pdays, or missing (NaN) where pdays == 999.
    This function learns nothing from the data, so it cannot leak. The imputation of the missing
    `pdays_days` values happens later in the pipeline, with a median learned from training rows only.
    """
    X = X.copy()
    is_999 = X['pdays'] == PDAYS_SENTINEL
    X['pdays_is_999'] = is_999.astype(int)
    X['pdays_days'] = X['pdays'].where(~is_999)
    return X.drop(columns=['pdays'])


def build_preprocessor():
    """Unfitted preprocessing pipeline: pdays handling, then imputing, scaling and one-hot encoding.

    - age: standardised
    - pdays_days: median imputation only
    - previous, pdays_is_999: passed through unchanged
    - categorical (7 columns): one-hot; "unknown" is an ordinary category; categories never seen in
      training are encoded as all zeros instead of raising an error

    Why `pdays_days` and `previous` are NOT standardised (decision of 2026-10-27): in the training rows only
    6 of 24,712 rows have a real `pdays` value and `previous` never exceeds 1, so both columns have almost no
    variation there. A scaler fitted on them gets a tiny spread (about 0.016 for `pdays_days`), and later rows
    would then be scaled to extreme values (up to about +-513). Leaving them unscaled avoids that instability.
    Caution: `pdays_days` rests on only 6 real training observations, so whatever relationship a model learns
    for it may be unreliable.
    """
    columns = ColumnTransformer([
        ('scaled', StandardScaler(), SCALED_COLUMNS),
        ('imputed', SimpleImputer(strategy='median'), IMPUTED_COLUMNS),
        ('passthrough', 'passthrough', PASSTHROUGH_COLUMNS),
        ('categorical', OneHotEncoder(handle_unknown='ignore'), CATEGORICAL_COLUMNS),
    ], remainder='drop')
    return Pipeline([('pdays', FunctionTransformer(add_pdays_features)), ('columns', columns)])
