"""The four data streams (scenarios) that every retraining strategy is run on.

Every scenario is returned as a `Scenario` with the same fields, so the strategy loop can treat them alike:

    initial_train   rows the first model is fitted on (preprocessing too)
    windows         list of DataFrames, the stream in arrival order (window 1 first)
    change_window   0-based index of the first shifted window, or None when no change point is known
    kind            'real' or 'synthetic'
    description     one sentence for tables and plots

1. real_replay   REAL data. The reserved last 20% of rows (8,238), cut into N_STREAM_WINDOWS consecutive
                 windows by row index. The initial model is fitted on the training rows. No known change point.
2. stationary    SYNTHETIC. Every window is drawn at random from one fixed pool, so nothing changes. Used to
                 count false alarms and needless retrains.
3. sudden        SYNTHETIC. Same as stationary until `SUDDEN_CHANGE_WINDOW`, then the distribution of
                 SHIFT_FEATURE jumps and stays shifted.
4. gradual       SYNTHETIC. The same shift, growing step by step from GRADUAL_START_WINDOW to GRADUAL_END_WINDOW.

How the synthetic scenarios are built (documented resampling of observed rows, AGENTS.md):
- The training rows are split once, at random with the fixed project SEED, into THREE disjoint parts
  (`synthetic_split`): a fitting part (`initial_train`, 60%), a scoring pool (20%, SYNTHETIC_POOL_FRAC) and a
  history pool (20%). Each window the models are SCORED on is a random sample (with replacement) of whole
  scoring-pool rows. The rows a retrain may LEARN from for that window (`Scenario.labelled_windows`) are a
  second, independent sample drawn the same way, with the same shift weights, from the history pool.
  So a scored row is never in any model's training data (the earlier design drew both from one small pool,
  and by window 8 most scored rows had already been used for retraining).
- Simulation assumption (document it in the report): the labels that arrive for window i come from a fresh
  sample of window i's distribution, not from the same customers that were scored.
- A row's features and its label always stay together. No values are invented.
- A covariate shift is made by sampling rows with unequal weights:
      weight = exp(strength * z),   z = (SHIFT_FEATURE - pool mean) / pool std
  With strength > 0, rows with a higher SHIFT_FEATURE value (older customers, for `age`) are drawn more often.
  Because whole rows are resampled, the relationship between features and label inside each row is unchanged
  (covariate shift), but the share of "yes" in a window can change if the shifted feature is related to it.
- These streams are controlled experiments, not historical evidence.
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.data import chronological_split
from src.settings import (GRADUAL_END_WINDOW, GRADUAL_START_WINDOW, N_STREAM_WINDOWS, SEED,
                          SHIFT_FEATURE, SHIFT_STRENGTH, SUDDEN_CHANGE_WINDOW, SYNTHETIC_POOL_FRAC,
                          SYNTHETIC_WINDOW_ROWS)


@dataclass
class Scenario:
    name: str
    kind: str                       # 'real' or 'synthetic'
    initial_train: pd.DataFrame
    windows: list
    change_window: int = None       # 0-based index of the first shifted window; None = unknown / no change
    description: str = ''
    strengths: list = field(default_factory=list)   # shift strength used for each window (synthetic only)
    labelled_windows: list = None   # rows whose labels arrive for each window (retraining data);
                                    # None = the scored windows themselves (real replay)


# ---------------------------------------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------------------------------------
def split_into_windows(rows, n_windows=N_STREAM_WINDOWS):
    """Cut rows into `n_windows` consecutive blocks in their existing order (sizes differ by at most 1)."""
    edges = np.linspace(0, len(rows), n_windows + 1).astype(int)
    return [rows.iloc[edges[i]:edges[i + 1]].copy() for i in range(n_windows)]


def synthetic_split(train_rows, pool_frac=SYNTHETIC_POOL_FRAC, seed=SEED):
    """Split training rows at random into three DISJOINT parts: (fit, scoring_pool, history_pool).

    Each part keeps the original row order (the cut-off tuning uses "the most recent rows"). Fixed seed: the
    same split every time. With the default 0.2: 14,828 fitting rows and two pools of 4,942 rows.
    """
    if not 0 < 2 * pool_frac < 1:
        raise ValueError('two pools must leave some rows for fitting')
    order = np.random.default_rng(seed).permutation(len(train_rows))
    n_pool = int(round(len(train_rows) * pool_frac))
    scoring = np.sort(order[:n_pool])
    history = np.sort(order[n_pool:2 * n_pool])
    fit = np.sort(order[2 * n_pool:])
    return train_rows.iloc[fit].copy(), train_rows.iloc[scoring].copy(), train_rows.iloc[history].copy()


def shift_weights(pool, strength, feature=SHIFT_FEATURE):
    """Sampling probabilities that tilt `feature` upwards (strength > 0) or leave it alone (strength = 0)."""
    values = pool[feature].to_numpy(float)
    z = (values - values.mean()) / values.std()
    weights = np.exp(strength * z)
    return weights / weights.sum()


def sample_window(pool, rows, rng, strength=0.0, feature=SHIFT_FEATURE):
    """Draw `rows` whole rows from the pool with replacement, shifted by `strength` (0 = no shift)."""
    probabilities = shift_weights(pool, strength, feature) if strength else None
    positions = rng.choice(len(pool), size=rows, replace=True, p=probabilities)
    return pool.iloc[positions].copy()


def gradual_strengths(n_windows, start, end, strength):
    """0 before `start`, rising linearly to `strength` at `end`, then staying at `strength`."""
    out = []
    for i in range(n_windows):
        if i < start:
            out.append(0.0)
        elif i >= end:
            out.append(float(strength))
        else:
            out.append(float(strength) * (i - start + 1) / (end - start + 1))
    return out


# ---------------------------------------------------------------------------------------------------------
# The four scenarios
# ---------------------------------------------------------------------------------------------------------
def real_replay(df, n_windows=N_STREAM_WINDOWS):
    """Scenario 1: the real stream in original order. Uses the confirmed 60/20/20 chronological split."""
    train, _dev, stream = chronological_split(df)
    return Scenario(name='real_replay', kind='real', initial_train=train,
                    windows=split_into_windows(stream, n_windows), change_window=None,
                    description=f'Real last 20% of rows in original order, {n_windows} windows')


def _synthetic(name, df, strengths, change_window, description, seed, window_rows, feature):
    train, _dev, _stream = chronological_split(df)
    fit_part, scoring_pool, history_pool = synthetic_split(train)     # fixed split (project SEED)
    # Per-run randomness (one of SEEDS): two independent generators, one for the scored windows and one for
    # their labelled copies, so the two sets of draws never depend on each other.
    score_rng, label_rng = (np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(2))
    windows = [sample_window(scoring_pool, window_rows, score_rng, s, feature) for s in strengths]
    labelled = [sample_window(history_pool, window_rows, label_rng, s, feature) for s in strengths]
    return Scenario(name=name, kind='synthetic', initial_train=fit_part, windows=windows,
                    change_window=change_window, description=description, strengths=list(strengths),
                    labelled_windows=labelled)


def stationary(df, seed=SEED, n_windows=N_STREAM_WINDOWS, window_rows=SYNTHETIC_WINDOW_ROWS):
    """Scenario 2: no change at all. Every window comes from the same pool with equal weights."""
    return _synthetic('stationary', df, [0.0] * n_windows, None,
                      'Synthetic: every window drawn from one fixed pool (no change)',
                      seed, window_rows, SHIFT_FEATURE)


def sudden(df, seed=SEED, n_windows=N_STREAM_WINDOWS, window_rows=SYNTHETIC_WINDOW_ROWS,
           change_window=SUDDEN_CHANGE_WINDOW, strength=SHIFT_STRENGTH, feature=SHIFT_FEATURE):
    """Scenario 3: `feature` shifts at once at `change_window` (0-based) and stays shifted."""
    strengths = [0.0 if i < change_window else float(strength) for i in range(n_windows)]
    return _synthetic('sudden', df, strengths, change_window,
                      f'Synthetic: {feature} shifts at window {change_window + 1} and stays shifted',
                      seed, window_rows, feature)


def gradual(df, seed=SEED, n_windows=N_STREAM_WINDOWS, window_rows=SYNTHETIC_WINDOW_ROWS,
            start=GRADUAL_START_WINDOW, end=GRADUAL_END_WINDOW, strength=SHIFT_STRENGTH, feature=SHIFT_FEATURE):
    """Scenario 4: the same shift, growing linearly from window `start` to full strength at window `end`."""
    return _synthetic('gradual', df, gradual_strengths(n_windows, start, end, strength), start,
                      f'Synthetic: {feature} shifts gradually from window {start + 1} to window {end + 1}',
                      seed, window_rows, feature)


def all_scenarios(df, seed=SEED):
    """All four scenarios for one seed, in the order used in tables: real, stationary, sudden, gradual.

    The real replay does not depend on the seed (it is the real data in order).
    """
    return [real_replay(df), stationary(df, seed), sudden(df, seed), gradual(df, seed)]
