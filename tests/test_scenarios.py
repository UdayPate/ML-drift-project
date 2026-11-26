"""Checks for src/scenarios.py on the real file: ordering, no leakage into evaluation, and the intended shifts.

Run from the project root:   python -m unittest discover -s tests -v
"""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import chronological_split, load_dataset
from src.drift import psi
from src.scenarios import (all_scenarios, gradual, gradual_strengths, real_replay, stationary, sudden,
                           synthetic_train_and_pool)
from src.settings import (GRADUAL_END_WINDOW, GRADUAL_START_WINDOW, N_STREAM_WINDOWS, PSI_ALERT, SHIFT_FEATURE,
                          SUDDEN_CHANGE_WINDOW, SYNTHETIC_WINDOW_ROWS)

TRAIN_END, DEV_END, N_ROWS = 24_712, 32_950, 41_188


class ScenarioTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = load_dataset()
        cls.train, cls.dev, cls.stream = chronological_split(cls.df)

    # Real replay -------------------------------------------------------------------------------------
    def test_real_replay_keeps_original_order_and_covers_stream_exactly(self):
        scenario = real_replay(self.df)
        self.assertEqual(len(scenario.windows), N_STREAM_WINDOWS)
        positions = np.concatenate([w.index.to_numpy() for w in scenario.windows])
        np.testing.assert_array_equal(positions, np.arange(DEV_END, N_ROWS))     # in order, no gaps, no repeats
        self.assertEqual(list(scenario.initial_train.index), list(range(TRAIN_END)))
        sizes = [len(w) for w in scenario.windows]
        self.assertLessEqual(max(sizes) - min(sizes), 1)
        self.assertIsNone(scenario.change_window)

    def test_real_replay_never_uses_dev_rows(self):
        scenario = real_replay(self.df)
        used = set(scenario.initial_train.index).union(*[set(w.index) for w in scenario.windows])
        self.assertTrue(used.isdisjoint(set(self.dev.index)))

    # Synthetic scenarios -----------------------------------------------------------------------------
    def test_pool_is_held_out_from_fitting_rows_and_split_is_fixed(self):
        fit_a, pool_a = synthetic_train_and_pool(self.train)
        fit_b, pool_b = synthetic_train_and_pool(self.train)
        self.assertTrue(set(fit_a.index).isdisjoint(set(pool_a.index)))
        self.assertEqual(len(fit_a) + len(pool_a), len(self.train))
        self.assertEqual(list(pool_a.index), list(pool_b.index))               # same split every time

    def test_synthetic_windows_only_use_pool_rows(self):
        _fit, pool = synthetic_train_and_pool(self.train)
        for scenario in (stationary(self.df, seed=42), sudden(self.df, seed=42), gradual(self.df, seed=42)):
            self.assertEqual(len(scenario.windows), N_STREAM_WINDOWS)
            for window in scenario.windows:
                self.assertEqual(len(window), SYNTHETIC_WINDOW_ROWS)
                self.assertTrue(set(window.index) <= set(pool.index))
                self.assertTrue(set(window.index).isdisjoint(set(scenario.initial_train.index)))

    def test_same_seed_same_stream_different_seed_different_stream(self):
        a, b, c = sudden(self.df, seed=42), sudden(self.df, seed=42), sudden(self.df, seed=43)
        self.assertEqual(list(a.windows[0].index), list(b.windows[0].index))
        self.assertNotEqual(list(a.windows[0].index), list(c.windows[0].index))

    def test_stationary_stays_below_alert(self):
        scenario = stationary(self.df, seed=42)
        for window in scenario.windows:
            self.assertLess(psi(scenario.initial_train[SHIFT_FEATURE], window[SHIFT_FEATURE]), PSI_ALERT)

    def test_sudden_shift_appears_exactly_at_change_window(self):
        scenario = sudden(self.df, seed=42)
        self.assertEqual(scenario.change_window, SUDDEN_CHANGE_WINDOW)
        values = [psi(scenario.initial_train[SHIFT_FEATURE], w[SHIFT_FEATURE]) for w in scenario.windows]
        self.assertTrue(all(v < PSI_ALERT for v in values[:SUDDEN_CHANGE_WINDOW]))
        self.assertTrue(all(v > PSI_ALERT for v in values[SUDDEN_CHANGE_WINDOW:]))

    def test_gradual_shift_grows(self):
        strengths = gradual_strengths(N_STREAM_WINDOWS, GRADUAL_START_WINDOW, GRADUAL_END_WINDOW, 1.0)
        self.assertEqual(strengths[:GRADUAL_START_WINDOW], [0.0] * GRADUAL_START_WINDOW)
        self.assertEqual(strengths[GRADUAL_END_WINDOW:], [1.0] * (N_STREAM_WINDOWS - GRADUAL_END_WINDOW))
        self.assertEqual(strengths, sorted(strengths))
        scenario = gradual(self.df, seed=42)
        means = [w[SHIFT_FEATURE].mean() for w in scenario.windows]
        self.assertLess(means[0], means[-1])
        self.assertGreater(psi(scenario.initial_train[SHIFT_FEATURE], scenario.windows[-1][SHIFT_FEATURE]),
                           PSI_ALERT)

    def test_all_scenarios_come_in_table_order(self):
        names = [s.name for s in all_scenarios(self.df, seed=42)]
        self.assertEqual(names, ['real_replay', 'stationary', 'sudden', 'gradual'])


if __name__ == '__main__':
    unittest.main()
