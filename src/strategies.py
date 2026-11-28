"""The three retraining strategies and the ONE shared loop that runs them on a scenario (Person 3).

Strategies
----------
- static           fit once on the scenario's initial training rows, never retrain
- periodic         retrain every PERIODIC_EVERY_WINDOWS windows
- drift_triggered  retrain when Person 1's drift_trigger() fires on the incoming window

Fair comparison (TA feedback #3)
--------------------------------
All three strategies go through the same function, `run_strategy`. They receive the same scenario object (same
windows, same rows), the same model configuration and features, the same seed, the same label-delay rule, the
same retraining data rule and the same threshold rule. The ONLY thing that differs is `should_retrain()`, the
answer to "do we refit before scoring this window?".

Timeline of one window i (0-based), the same for every strategy
--------------------------------------------------------------
1. Window i arrives. Its FEATURES are visible (they are known before the call); its labels are not.
2. Drift monitoring: drift_trigger(rows the current model was trained on, window i). Logged for EVERY strategy,
   so alerts and false alarms can be compared; only drift_triggered acts on it.
3. Retrain decision. A retrain may only use labelled rows: the initial training rows plus stream windows
   0 .. i-1-LABEL_DELAY_WINDOWS (with a delay of 1: before window i, windows up to i-2 are labelled).
   Never the current or a future window (AGENTS.md). If no new labelled rows arrived since the last fit, there
   is nothing new to learn from and the retrain is skipped (logged as `retrain_skipped_no_new_labels`).
4. Window i is scored with the current model and threshold; metrics are computed against its true labels.
5. Window i's labels become available LABEL_DELAY_WINDOWS windows later.

Every fit (initial and retrain) follows `fit_model`: preprocessing and model are refitted together on exactly
the same rows, and the decision threshold is re-tuned on the most recent labelled rows (see settings).
"""
import time

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.pipeline import Pipeline

from src.data import encode_target
from src.drift import drift_trigger
from src.metrics import evaluate_binary
from src.model_comparison import (FEATURE_SETS, best_f1_threshold, model_candidates, positive_scores,
                                  preprocessor_for)
from src.settings import (EXPERIMENT_FEATURE_SET, EXPERIMENT_MODELS, LABEL_DELAY_WINDOWS,
                          PERIODIC_EVERY_WINDOWS, RETRAIN_DATA, SEED, STRATEGIES, THRESHOLD_MIN_POSITIVES,
                          THRESHOLD_TUNE_ROWS)


# ---------------------------------------------------------------------------------------------------------
# Model construction and fitting
# ---------------------------------------------------------------------------------------------------------
def make_pipeline(model_name, seed=SEED, feature_set=EXPERIMENT_FEATURE_SET):
    """Unfitted preprocessing + model, using Person 2's selected configuration for `model_name`."""
    config_index = EXPERIMENT_MODELS[model_name]
    estimator = clone(model_candidates()[model_name][config_index])
    if 'random_state' in estimator.get_params():
        estimator.set_params(random_state=seed)
    return Pipeline([('prep', preprocessor_for(feature_set)), ('model', estimator)])


def fit_model(model_name, labelled_rows, seed=SEED, previous_threshold=None,
              feature_set=EXPERIMENT_FEATURE_SET, tune_rows=THRESHOLD_TUNE_ROWS,
              min_positives=THRESHOLD_MIN_POSITIVES):
    """Fit preprocessing + model on `labelled_rows` and choose a decision threshold. Used for EVERY fit.

    `labelled_rows` must be in arrival order (oldest first). Threshold rule:
      1. hold out the most recent `tune_rows` rows, fit a temporary model on the older rows,
      2. pick the cut-off with the best F1 on the held-out rows (Person 2's best_f1_threshold),
      3. refit on ALL labelled rows and keep that cut-off.
    If the held-out rows have fewer than `min_positives` "yes" rows the cut-off is unreliable, so
    `previous_threshold` is kept (if there is one).

    Returns (fitted pipeline, threshold, info dict with fit time and row counts).
    """
    features = FEATURE_SETS[feature_set]
    X, y = labelled_rows[features], encode_target(labelled_rows).to_numpy()

    start = time.perf_counter()
    older, recent = slice(0, len(X) - tune_rows), slice(len(X) - tune_rows, len(X))
    threshold, threshold_source = previous_threshold, 'kept_previous'
    if y[recent].sum() >= min_positives and len(set(y[older])) == 2:
        temporary = make_pipeline(model_name, seed, feature_set).fit(X.iloc[older], y[older])
        scores = positive_scores(temporary, X.iloc[recent])
        if len(np.unique(scores)) > 1:
            threshold, _ = best_f1_threshold(y[recent], scores)
            threshold_source = 'tuned_on_recent_rows'
    if threshold is None:
        raise ValueError('Could not tune a threshold and there is no previous one to keep')

    pipeline = make_pipeline(model_name, seed, feature_set).fit(X, y)
    seconds = time.perf_counter() - start
    info = {'fit_seconds': seconds, 'n_train_rows': len(X), 'n_train_positive': int(y.sum()),
            'threshold_source': threshold_source}
    return pipeline, float(threshold), info


# ---------------------------------------------------------------------------------------------------------
# Labelled data and the retraining rules
# ---------------------------------------------------------------------------------------------------------
def n_labelled_windows(window_index, label_delay=LABEL_DELAY_WINDOWS):
    """How many stream windows have labels just before window `window_index` (0-based) is scored.

    Window j's labels arrive `label_delay` windows after window j has been scored, so before window i the
    windows 0 .. i-1-label_delay are labelled. With label_delay = 0 the previous window is already labelled.
    """
    return max(0, window_index - label_delay)


def labelled_history(scenario, window_index, label_delay=LABEL_DELAY_WINDOWS, retrain_data=RETRAIN_DATA):
    """Rows a model may be (re)trained on just before window `window_index`: oldest first."""
    if retrain_data != 'expanding':
        raise NotImplementedError(f"Only 'expanding' retraining data is implemented, got {retrain_data!r}")
    # Synthetic scenarios carry separate labelled copies of each window (never the scored rows themselves);
    # the real replay has none, so its scored windows are its labelled windows.
    source = scenario.windows if scenario.labelled_windows is None else scenario.labelled_windows
    labelled = source[:n_labelled_windows(window_index, label_delay)]
    return pd.concat([scenario.initial_train, *labelled])


def should_retrain(strategy, window_index, drift_decision, periodic_every=PERIODIC_EVERY_WINDOWS):
    """The ONLY place where the three strategies differ: retrain before scoring this window?"""
    if strategy == 'static':
        return False
    if strategy == 'periodic':
        return window_index > 0 and window_index % periodic_every == 0
    if strategy == 'drift_triggered':
        return bool(drift_decision['retrain'])
    raise ValueError(f'Unknown strategy {strategy!r}; expected one of {STRATEGIES}')


# ---------------------------------------------------------------------------------------------------------
# The shared loop
# ---------------------------------------------------------------------------------------------------------
def run_strategy(scenario, strategy, model_name, seed=SEED, label_delay=LABEL_DELAY_WINDOWS,
                 periodic_every=PERIODIC_EVERY_WINDOWS, initial_fit=None, feature_set=EXPERIMENT_FEATURE_SET,
                 scores_out=None):
    """Run one strategy on one scenario with one model and seed. Returns one row per window (a DataFrame).

    `initial_fit` lets a caller pass an already fitted (pipeline, threshold, info) for the initial model, so the
    three strategies on the same scenario/model/seed start from the IDENTICAL model (and it is fitted once).

    `scores_out` (optional, default None = nothing recorded, results unchanged): pass an empty list and, for every
    window, a dict with the strategy, 1-based window, the window's row ids, true labels, scores and the threshold
    used is appended to it. Used only for diagnostics that need per-row scores (bootstrap, error analysis).
    """
    features = FEATURE_SETS[feature_set]
    if initial_fit is None:
        initial_fit = fit_model(model_name, scenario.initial_train, seed, feature_set=feature_set)
    model, threshold, fit_info = initial_fit
    reference = scenario.initial_train            # rows the current model was trained on (drift reference)
    labelled_windows_in_model = 0
    model_version = 0

    rows = []
    for i, window in enumerate(scenario.windows):
        # 2. Drift monitoring on the incoming window's features (labels are not used).
        start = time.perf_counter()
        decision = drift_trigger(reference, window)
        monitor_seconds = time.perf_counter() - start
        report = decision['report']
        economic = report[report['group'] == 'economic']

        # 3. Retrain decision (only `should_retrain` depends on the strategy).
        wants_retrain = should_retrain(strategy, i, decision, periodic_every)
        available = n_labelled_windows(i, label_delay)
        has_new_labels = available > labelled_windows_in_model
        retrained, retrain_seconds, threshold_source = False, 0.0, ''
        if wants_retrain and has_new_labels:
            history = labelled_history(scenario, i, label_delay)
            model, threshold, fit_info = fit_model(model_name, history, seed, previous_threshold=threshold,
                                                   feature_set=feature_set)
            reference = history
            labelled_windows_in_model = available
            model_version += 1
            retrained, retrain_seconds = True, fit_info['fit_seconds']
            threshold_source = fit_info['threshold_source']

        # 4. Score the window and evaluate against its true labels.
        y = encode_target(window).to_numpy()
        scores = positive_scores(model, window[features])
        if len(set(y)) < 2:   # never happens on our streams (every window has dozens of "yes" rows)
            raise ValueError(f'{scenario.name} window {i + 1} has only one class; PR-AUC is undefined')
        metrics = evaluate_binary(y, scores, threshold=threshold)
        if scores_out is not None:
            scores_out.append({'strategy': strategy, 'window': i + 1, 'index': window.index.to_numpy(),
                               'y': y, 'scores': scores, 'threshold': float(threshold)})

        rows.append({
            'scenario': scenario.name, 'scenario_kind': scenario.kind, 'strategy': strategy,
            'model': model_name, 'feature_set': feature_set, 'seed': seed,
            'window': i + 1,                                     # 1-based for tables and plots
            'change_window': None if scenario.change_window is None else scenario.change_window + 1,
            'after_change': (scenario.change_window is not None and i >= scenario.change_window),
            'n_rows': metrics['n_rows'], 'n_positive': metrics['n_positive'],
            'pr_auc': metrics['pr_auc'], 'random_pr_auc': metrics['random_pr_auc'],
            'roc_auc': metrics['roc_auc'], 'f1': metrics['f1'], 'precision': metrics['precision'],
            'recall': metrics['recall'], 'tp': metrics['tp'], 'fp': metrics['fp'], 'fn': metrics['fn'],
            'tn': metrics['tn'], 'n_predicted_positive': metrics['n_predicted_positive'],
            'threshold': threshold,
            'alert': bool(decision['retrain']),                  # trigger rule fired (logged for all strategies)
            'alert_features': ';'.join(decision['fired']),
            'max_psi_trigger_groups': decision['max_psi'],
            'economic_alert': bool(economic['alert'].any()),     # logged separately; does not trigger
            'max_psi_economic': float(economic['psi'].max()) if len(economic) else float('nan'),
            'wanted_retrain': wants_retrain,
            'retrained': retrained,
            'retrain_skipped_no_new_labels': wants_retrain and not has_new_labels,
            'retrain_seconds': retrain_seconds,
            'threshold_source': threshold_source,
            'model_version': model_version,
            'n_train_rows': len(reference),
            'labelled_windows_in_model': labelled_windows_in_model,
            'monitor_seconds': monitor_seconds,
        })
    return pd.DataFrame(rows)


def run_all_strategies(scenario, model_name, seed=SEED, strategies=STRATEGIES, **kwargs):
    """All strategies on one scenario/model/seed, starting from ONE shared initial model."""
    initial_fit = fit_model(model_name, scenario.initial_train, seed,
                            feature_set=kwargs.get('feature_set', EXPERIMENT_FEATURE_SET))
    frames = []
    for strategy in strategies:
        result = run_strategy(scenario, strategy, model_name, seed, initial_fit=initial_fit, **kwargs)
        result['initial_fit_seconds'] = initial_fit[2]['fit_seconds']
        result['initial_threshold'] = initial_fit[1]
        frames.append(result)
    return pd.concat(frames, ignore_index=True)


def run_experiment(df, seeds, models=None, scenario_builder=None, strategies=STRATEGIES, verbose=True):
    """The full grid: scenarios x strategies x models x seeds. Returns one row per window.

    `scenario_builder(df, seed)` returns the list of scenarios for a seed (default: all_scenarios).
    """
    if scenario_builder is None:
        from src.scenarios import all_scenarios
        scenario_builder = all_scenarios
    models = list(EXPERIMENT_MODELS) if models is None else models
    frames = []
    for seed in seeds:
        for scenario in scenario_builder(df, seed=seed):
            for model_name in models:
                start = time.perf_counter()
                frames.append(run_all_strategies(scenario, model_name, seed, strategies))
                if verbose:
                    print(f'seed {seed}  {scenario.name:12s} {model_name:20s} '
                          f'done in {time.perf_counter() - start:5.1f} s')
    return pd.concat(frames, ignore_index=True)


def summarise(results):
    """One row per scenario/strategy/model (averaged over seeds): a quick sanity table, not the final analysis."""
    per_run = (results.groupby(['scenario', 'strategy', 'model', 'seed'])
               .agg(mean_pr_auc=('pr_auc', 'mean'), mean_random_pr_auc=('random_pr_auc', 'mean'),
                    mean_f1=('f1', 'mean'), retrains=('retrained', 'sum'), alerts=('alert', 'sum'),
                    retrain_seconds=('retrain_seconds', 'sum'))
               .reset_index())
    return (per_run.groupby(['scenario', 'strategy', 'model'])
            .agg(mean_pr_auc=('mean_pr_auc', 'mean'), sd_pr_auc_over_seeds=('mean_pr_auc', 'std'),
                 mean_random_pr_auc=('mean_random_pr_auc', 'mean'), mean_f1=('mean_f1', 'mean'),
                 retrains=('retrains', 'mean'), alerts=('alerts', 'mean'),
                 retrain_seconds=('retrain_seconds', 'mean'))
            .reset_index())
