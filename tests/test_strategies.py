"""Checks for src/strategies.py: label delay, no future rows in retraining, and a fair shared loop.

These tests build small fake streams, so they run without the dataset file.
Run from the project root:   python -m unittest discover -s tests -v
"""
import sys
import unittest
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.scenarios import Scenario
from src.settings import ECONOMIC_FEATURES, STRATEGIES
from src.strategies import (fit_model, labelled_history, n_labelled_windows, run_all_strategies,
                            should_retrain)

MODEL = 'Logistic Regression'


def fake_rows(n, rng, start_index, yes_rate=0.15, age_shift=0.0):
    """Fake rows with every column the drift monitor and the model need (values are made up)."""
    age = np.clip(rng.normal(40 + age_shift, 10, n), 18, 95).round()
    previous = rng.integers(0, 2, n)
    poutcome = np.where(previous == 0, 'nonexistent', rng.choice(['failure', 'success'], n))
    signal = 0.04 * (age - 40) + 1.5 * (poutcome == 'success')
    p_yes = 1 / (1 + np.exp(-(np.log(yes_rate / (1 - yes_rate)) + signal)))
    frame = pd.DataFrame({
        'age': age,
        'job': rng.choice(['admin.', 'technician', 'retired', 'student'], n),
        'marital': rng.choice(['married', 'single', 'divorced'], n),
        'education': rng.choice(['high.school', 'university.degree', 'unknown'], n),
        'default': rng.choice(['no', 'unknown'], n),
        'housing': rng.choice(['yes', 'no'], n),
        'loan': rng.choice(['yes', 'no'], n),
        'pdays': np.where(previous > 0, rng.integers(0, 15, n), 999),
        'previous': previous,
        'poutcome': poutcome,
        'duration': rng.integers(0, 500, n),
        'y': np.where(rng.random(n) < p_yes, 'yes', 'no'),
    }, index=np.arange(start_index, start_index + n))
    for column in ECONOMIC_FEATURES:
        frame[column] = rng.normal(0, 1, n).round(2)
    return frame


def fake_scenario(n_windows=6, window_rows=400, shift_from=None, seed=0):
    """Initial training rows plus `n_windows` windows; from `shift_from` on, age is shifted strongly."""
    rng = np.random.default_rng(seed)
    initial = fake_rows(3000, rng, 0)
    windows = []
    for i in range(n_windows):
        shift = 25.0 if (shift_from is not None and i >= shift_from) else 0.0
        windows.append(fake_rows(window_rows, rng, 3000 + i * window_rows, age_shift=shift))
    return Scenario(name='fake', kind='synthetic', initial_train=initial, windows=windows,
                    change_window=shift_from, description='fake test stream')


class LabelDelayTests(unittest.TestCase):
    def test_number_of_labelled_windows(self):
        # delay 1: before window i (0-based), windows 0..i-2 are labelled
        self.assertEqual([n_labelled_windows(i, 1) for i in range(5)], [0, 0, 1, 2, 3])
        self.assertEqual([n_labelled_windows(i, 0) for i in range(4)], [0, 1, 2, 3])
        self.assertEqual([n_labelled_windows(i, 2) for i in range(5)], [0, 0, 0, 1, 2])

    def test_history_never_contains_current_or_future_windows(self):
        scenario = fake_scenario()
        for delay in (0, 1, 2):
            for i in range(len(scenario.windows)):
                history = labelled_history(scenario, i, label_delay=delay)
                forbidden = pd.concat(scenario.windows[max(0, i - delay):])     # current, future, not-yet-labelled
                self.assertFalse(set(history.index) & set(forbidden.index), (delay, i))
                self.assertTrue(set(scenario.initial_train.index) <= set(history.index))

    def test_history_is_in_arrival_order(self):
        scenario = fake_scenario()
        history = labelled_history(scenario, 5, label_delay=1)
        self.assertTrue(history.index.is_monotonic_increasing)


class RetrainRuleTests(unittest.TestCase):
    def test_static_never_retrains(self):
        for i in range(8):
            self.assertFalse(should_retrain('static', i, {'retrain': True}))

    def test_periodic_every_two_windows(self):
        wanted = [i for i in range(8) if should_retrain('periodic', i, {'retrain': False}, periodic_every=2)]
        self.assertEqual(wanted, [2, 4, 6])

    def test_drift_triggered_follows_the_trigger(self):
        self.assertTrue(should_retrain('drift_triggered', 3, {'retrain': True}))
        self.assertFalse(should_retrain('drift_triggered', 3, {'retrain': False}))

    def test_unknown_strategy_is_rejected(self):
        with self.assertRaises(ValueError):
            should_retrain('sometimes', 0, {'retrain': False})


class SharedLoopTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        warnings.simplefilter('ignore')
        cls.scenario = fake_scenario(shift_from=3)
        cls.results = run_all_strategies(cls.scenario, MODEL, seed=42)

    def by_strategy(self, strategy):
        return self.results[self.results['strategy'] == strategy].reset_index(drop=True)

    def test_every_strategy_sees_every_window_once(self):
        for strategy in STRATEGIES:
            part = self.by_strategy(strategy)
            self.assertEqual(list(part['window']), list(range(1, len(self.scenario.windows) + 1)))
            self.assertEqual(list(part['n_rows']), [len(w) for w in self.scenario.windows])

    def test_all_strategies_start_from_the_same_model(self):
        first = self.results[self.results['window'] == 1]
        self.assertEqual(first['threshold'].nunique(), 1)
        self.assertEqual(first['pr_auc'].nunique(), 1)
        self.assertEqual(first['alert'].nunique(), 1)

    def test_identical_results_until_the_first_retrain(self):
        static = self.by_strategy('static')
        for strategy in ('periodic', 'drift_triggered'):
            part = self.by_strategy(strategy)
            first_retrain = part.index[part['retrained']].min()
            before = slice(0, first_retrain - 1) if not np.isnan(first_retrain) else slice(None)
            pd.testing.assert_series_equal(part.loc[before, 'pr_auc'], static.loc[before, 'pr_auc'])
            pd.testing.assert_series_equal(part.loc[before, 'f1'], static.loc[before, 'f1'])

    def test_static_never_changes_model(self):
        static = self.by_strategy('static')
        self.assertFalse(static['retrained'].any())
        self.assertEqual(static['model_version'].max(), 0)
        self.assertEqual(static['n_train_rows'].nunique(), 1)

    def test_periodic_retrains_on_schedule_with_the_right_rows(self):
        part = self.by_strategy('periodic')
        self.assertEqual(list(part.index[part['retrained']] + 1), [3, 5])        # windows 3 and 5 (1-based)
        row = part.iloc[4]                                                     # window 5: windows 1-3 labelled
        self.assertEqual(row['n_train_rows'], len(self.scenario.initial_train) + 3 * 400)

    def test_drift_trigger_reacts_to_the_shift_only(self):
        part = self.by_strategy('drift_triggered')
        self.assertFalse(part.loc[:2, 'alert'].any())                          # no alert before the shift
        self.assertTrue(part.loc[3:, 'alert'].all())                           # alert on every shifted window
        self.assertTrue(part.loc[3, 'retrained'])

    def test_retrain_skipped_when_no_new_labels(self):
        scenario = fake_scenario(shift_from=0, n_windows=3)
        part = run_all_strategies(scenario, MODEL, seed=42, strategies=['drift_triggered'])
        # windows 1 and 2: alert, but with a delay of 1 no stream labels exist yet
        self.assertEqual(list(part['retrain_skipped_no_new_labels']), [True, True, False])
        self.assertEqual(list(part['retrained']), [False, False, True])

    def test_logs_contain_the_required_columns(self):
        required = {'pr_auc', 'random_pr_auc', 'f1', 'precision', 'recall', 'alert', 'retrained',
                    'retrain_seconds', 'threshold', 'economic_alert', 'seed', 'model', 'scenario', 'strategy'}
        self.assertTrue(required <= set(self.results.columns))


class ThresholdTests(unittest.TestCase):
    def test_keeps_previous_threshold_when_too_few_positives(self):
        warnings.simplefilter('ignore')
        rows = fake_rows(2000, np.random.default_rng(1), 0)
        rows.iloc[-1030:, rows.columns.get_loc('y')] = 'no'                      # no "yes" in recent rows
        _model, threshold, info = fit_model(MODEL, rows, previous_threshold=0.123)
        self.assertEqual(threshold, 0.123)
        self.assertEqual(info['threshold_source'], 'kept_previous')

    def test_tunes_threshold_on_recent_rows(self):
        warnings.simplefilter('ignore')
        rows = fake_rows(3000, np.random.default_rng(2), 0)
        _model, threshold, info = fit_model(MODEL, rows)
        self.assertEqual(info['threshold_source'], 'tuned_on_recent_rows')
        self.assertTrue(0 < threshold < 1)
        self.assertEqual(info['n_train_rows'], 3000)


if __name__ == '__main__':
    unittest.main()
