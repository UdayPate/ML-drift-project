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
- The training rows are split once, at random with the fixed project SEED, into a fitting part
  (`initial_train`, 80%) and a held-out pool (20%, SYNTHETIC_POOL_FRAC). Pool rows are never used for fitting,
  so evaluation never reuses training records.
- Each window is a random sample (with replacement) of whole pool rows: a row's features and its label always
  stay together. No values are invented.
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


# ---------------------------------------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------------------------------------
def split_into_windows(rows, n_windows=N_STREAM_WINDOWS):
    """Cut rows into `n_windows` consecutive blocks in their existing order (sizes differ by at most 1)."""
    edges = np.linspace(0, len(rows), n_windows + 1).astype(int)
    return [rows.iloc[edges[i]:edges[i + 1]].copy() for i in range(n_windows)]


def synthetic_train_and_pool(train_rows, pool_frac=SYNTHETIC_POOL_FRAC, seed=SEED):
    """Split training rows at random into (fitting part, held-out pool). Fixed seed: same split every time."""
    rng = np.random.default_rng(seed)
    in_pool = np.zeros(len(train_rows), dtype=bool)
    in_pool[rng.choice(len(train_rows), size=int(round(len(train_rows) * pool_frac)), replace=False)] = True
    return train_rows.iloc[~in_pool].copy(), train_rows.iloc[in_pool].copy()


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
    fit_part, pool = synthetic_train_and_pool(train)           # fixed split (project SEED)
    rng = np.random.default_rng(seed)                           # per-run randomness (one of SEEDS)
    windows = [sample_window(pool, window_rows, rng, s, feature) for s in strengths]
    return Scenario(name=name, kind='synthetic', initial_train=fit_part, windows=windows,
                    change_window=change_window, description=description, strengths=list(strengths))


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
