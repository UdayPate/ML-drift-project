"""Drift detection: PSI and KS per feature, per window, and the drift-trigger rule.

Two measures are computed for every monitored feature, comparing a REFERENCE sample (normally the rows the
current model was trained on) with a CURRENT window:

- PSI (Population Stability Index): how much the share of rows in each bin moved. 0 = identical shares.
  Rule of thumb: below 0.1 small, 0.1-0.25 moderate, above 0.25 major. It is a rule of thumb, not a test.
- KS (Kolmogorov-Smirnov) statistic: the largest gap between the two cumulative distributions (0 to 1).
  Numeric features only. With thousands of rows almost any difference gives a tiny p-value, so the KS
  statistic (an effect size) matters more than its p-value. KS is reported, but it does not trigger retraining.

Neither measure says anything about model performance: an alert means "the inputs changed", not
"predictions got worse". Whether alerts line up with real performance drops is checked in the experiment.

Typical use inside a strategy loop:

    decision = drift_trigger(reference_rows, current_window)
    if decision['retrain']:
        ...
"""
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from src.settings import MONITOR_GROUPS, PSI_ALERT, PSI_BINS, PSI_EPS, SEED, TRIGGER_GROUPS


# ---------------------------------------------------------------------------------------------------------
# Single-feature measures
# ---------------------------------------------------------------------------------------------------------
def is_categorical(values, bins=PSI_BINS):
    """Treat text columns, and numeric columns with at most `bins` distinct values, as categories.

    Several features are numbers with only a handful of distinct values (e.g. `previous` has 2 distinct
    values in the training rows, `emp.var.rate` has 3). Quantile bins do not work on them, so each distinct
    value becomes its own bin instead.
    """
    values = pd.Series(values)
    return (not pd.api.types.is_numeric_dtype(values)) or values.nunique() <= bins


def _shares(counts, total, eps):
    return np.clip(counts / total, eps, None)


def psi(reference, current, bins=PSI_BINS, eps=PSI_EPS):
    """Population Stability Index between two samples of ONE feature.

    Categorical (or few-valued numeric) features: one bin per value seen in either sample.
    Numeric features: bin edges are quantiles of the REFERENCE sample only, so the current window never
    influences the bins; the outer edges are opened to -inf/+inf so every current value lands in a bin.
    Empty bins get the small floor `eps` instead of 0 (log(0) is undefined). The value depends on the binning
    and the floor, so compare PSI values computed the same way, not across different settings.
    """
    reference = pd.Series(reference).reset_index(drop=True)
    current = pd.Series(current).reset_index(drop=True)
    if len(reference) == 0 or len(current) == 0:
        raise ValueError('psi() needs non-empty reference and current samples')

    if is_categorical(reference, bins):
        categories = sorted(set(reference.astype(str)) | set(current.astype(str)))
        ref_counts = reference.astype(str).value_counts().reindex(categories, fill_value=0).to_numpy(float)
        cur_counts = current.astype(str).value_counts().reindex(categories, fill_value=0).to_numpy(float)
    else:
        edges = np.unique(np.quantile(reference.to_numpy(float), np.linspace(0, 1, bins + 1)))
        edges[0], edges[-1] = -np.inf, np.inf
        ref_counts = np.histogram(reference.to_numpy(float), edges)[0].astype(float)
        cur_counts = np.histogram(current.to_numpy(float), edges)[0].astype(float)

    ref_share = _shares(ref_counts, len(reference), eps)
    cur_share = _shares(cur_counts, len(current), eps)
    return float(np.sum((cur_share - ref_share) * np.log(cur_share / ref_share)))


def ks(reference, current):
    """KS statistic and p-value for a numeric feature; (nan, nan) for a text feature."""
    reference, current = pd.Series(reference), pd.Series(current)
    if not pd.api.types.is_numeric_dtype(reference):
        return float('nan'), float('nan')
    result = ks_2samp(reference.to_numpy(float), current.to_numpy(float))
    return float(result.statistic), float(result.pvalue)


# ---------------------------------------------------------------------------------------------------------
# One window: every monitored feature
# ---------------------------------------------------------------------------------------------------------
def drift_report(reference_df, current_df, groups=None, psi_alert=PSI_ALERT):
    """One row per monitored feature: group, PSI, KS statistic, KS p-value, and whether PSI > psi_alert.

    `groups` maps a group name to its feature list; it defaults to MONITOR_GROUPS from settings.
    """
    groups = MONITOR_GROUPS if groups is None else groups
    rows = []
    for group, features in groups.items():
        for feature in features:
            ks_stat, ks_p = ks(reference_df[feature], current_df[feature])
            value = psi(reference_df[feature], current_df[feature])
            rows.append({'group': group, 'feature': feature, 'psi': value,
                         'ks_stat': ks_stat, 'ks_p': ks_p, 'alert': value > psi_alert})
    return pd.DataFrame(rows)


def drift_trigger(reference_df, current_df, trigger_groups=None, psi_alert=PSI_ALERT, groups=None):
    """The PROPOSED retraining rule: retrain when ANY feature in the trigger groups has PSI > psi_alert.

    Returns a dict:
      retrain       True/False
      fired         list of trigger-group features whose PSI is above the threshold
      max_psi       the largest PSI among trigger-group features
      report        the full drift_report (all monitored groups, for logging)

    Economic indicators are still measured and logged even when they are not in `trigger_groups`, so
    their alerts can be reported separately.
    """
    trigger_groups = TRIGGER_GROUPS if trigger_groups is None else trigger_groups
    report = drift_report(reference_df, current_df, groups=groups, psi_alert=psi_alert)
    trigger_rows = report[report['group'].isin(trigger_groups)]
    fired = trigger_rows.loc[trigger_rows['alert'], 'feature'].tolist()
    return {'retrain': bool(fired), 'fired': fired,
            'max_psi': float(trigger_rows['psi'].max()) if len(trigger_rows) else float('nan'),
            'report': report}


# ---------------------------------------------------------------------------------------------------------
# Many windows, and threshold calibration
# ---------------------------------------------------------------------------------------------------------
def drift_over_windows(reference_df, windows, groups=None, psi_alert=PSI_ALERT):
    """drift_report for each window against one fixed reference, stacked with a `window` column (1-based)."""
    frames = []
    for number, window in enumerate(windows, start=1):
        report = drift_report(reference_df, window, groups=groups, psi_alert=psi_alert)
        report.insert(0, 'window', number)
        frames.append(report)
    return pd.concat(frames, ignore_index=True)


def null_psi_distribution(reference_df, window_rows, n_draws=200, groups=None, seed=SEED):
    """PSI values we get when NOTHING has changed, used to check the alert threshold.

    Each draw takes `window_rows` rows at random (without replacement) from the reference itself and computes
    PSI against the whole reference. Any PSI here is pure sampling noise, so the share of draws above the
    alert threshold estimates the false-alarm rate of the rule per window. Use training or development
    rows only, never stream rows, so the stream cannot influence the threshold.

    Returns one row per (draw, feature) with columns draw, group, feature, psi.
    """
    groups = MONITOR_GROUPS if groups is None else groups
    rng = np.random.default_rng(seed)
    rows = []
    for draw in range(n_draws):
        positions = rng.choice(len(reference_df), size=window_rows, replace=False)
        sample = reference_df.iloc[positions]
        for group, features in groups.items():
            for feature in features:
                rows.append({'draw': draw, 'group': group, 'feature': feature,
                             'psi': psi(reference_df[feature], sample[feature])})
    return pd.DataFrame(rows)


def false_alarm_rate(null_psi, psi_alert=PSI_ALERT, trigger_groups=None):
    """Share of no-change draws in which the trigger rule would still have fired."""
    trigger_groups = TRIGGER_GROUPS if trigger_groups is None else trigger_groups
    in_scope = null_psi[null_psi['group'].isin(trigger_groups)]
    fired_per_draw = in_scope.groupby('draw')['psi'].max() > psi_alert
    return float(fired_per_draw.mean())
