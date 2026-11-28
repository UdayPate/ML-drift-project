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
from src.scenarios import all_scenarios, gradual, gradual_strengths, real_replay, stationary, sudden, synthetic_split
from src.settings import (GRADUAL_END_WINDOW, GRADUAL_START_WINDOW, N_STREAM_WINDOWS, PSI_ALERT, SEEDS,
                          SHIFT_FEATURE, SUDDEN_CHANGE_WINDOW, SYNTHETIC_WINDOW_ROWS)

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
    def test_three_parts_are_disjoint_cover_the_training_rows_and_the_split_is_fixed(self):
        fit_a, scoring_a, history_a = synthetic_split(self.train)
        fit_b, scoring_b, history_b = synthetic_split(self.train)
        parts = [set(fit_a.index), set(scoring_a.index), set(history_a.index)]
        self.assertTrue(parts[0].isdisjoint(parts[1]) and parts[0].isdisjoint(parts[2]) and parts[1].isdisjoint(parts[2]))
        self.assertEqual(set().union(*parts), set(self.train.index))
        self.assertEqual((len(fit_a), len(scoring_a), len(history_a)), (14_828, 4_942, 4_942))
        for part in (fit_a, scoring_a, history_a):
            self.assertEqual(list(part.index), sorted(part.index))             # original row order kept
        self.assertEqual(list(scoring_a.index), list(scoring_b.index))        # same split every time
        self.assertEqual(list(history_a.index), list(history_b.index))

    def test_scored_windows_use_the_scoring_pool_and_labelled_windows_the_history_pool(self):
        fit, scoring, history = synthetic_split(self.train)
        for scenario in (stationary(self.df, seed=42), sudden(self.df, seed=42), gradual(self.df, seed=42)):
            self.assertEqual(list(scenario.initial_train.index), list(fit.index))
            self.assertEqual(len(scenario.windows), N_STREAM_WINDOWS)
            self.assertEqual(len(scenario.labelled_windows), N_STREAM_WINDOWS)
            for window, labelled in zip(scenario.windows, scenario.labelled_windows):
                self.assertEqual(len(window), SYNTHETIC_WINDOW_ROWS)
                self.assertEqual(len(labelled), SYNTHETIC_WINDOW_ROWS)
                self.assertTrue(set(window.index) <= set(scoring.index))
                self.assertTrue(set(labelled.index) <= set(history.index))

    def test_real_replay_has_no_separate_labelled_windows(self):
        self.assertIsNone(real_replay(self.df).labelled_windows)

    def test_same_seed_same_stream_different_seed_different_stream(self):
        a, b, c = sudden(self.df, seed=42), sudden(self.df, seed=42), sudden(self.df, seed=43)
        self.assertEqual(list(a.windows[0].index), list(b.windows[0].index))
        self.assertNotEqual(list(a.windows[0].index), list(c.windows[0].index))

    def test_stationary_stays_below_alert(self):
        for seed in SEEDS:
            scenario = stationary(self.df, seed=seed)
            for window in scenario.windows:
                self.assertLess(psi(scenario.initial_train[SHIFT_FEATURE], window[SHIFT_FEATURE]), PSI_ALERT)

    def test_sudden_shift_appears_exactly_at_change_window(self):
        for seed in SEEDS:
            scenario = sudden(self.df, seed=seed)
            self.assertEqual(scenario.change_window, SUDDEN_CHANGE_WINDOW)
            values = [psi(scenario.initial_train[SHIFT_FEATURE], w[SHIFT_FEATURE]) for w in scenario.windows]
            self.assertTrue(all(v < PSI_ALERT for v in values[:SUDDEN_CHANGE_WINDOW]))
            self.assertTrue(all(v > PSI_ALERT for v in values[SUDDEN_CHANGE_WINDOW:]))

    def test_gradual_shift_grows(self):
        strengths = gradual_strengths(N_STREAM_WINDOWS, GRADUAL_START_WINDOW, GRADUAL_END_WINDOW, 1.0)
        self.assertEqual(strengths[:GRADUAL_START_WINDOW], [0.0] * GRADUAL_START_WINDOW)
        self.assertEqual(strengths[GRADUAL_END_WINDOW:], [1.0] * (N_STREAM_WINDOWS - GRADUAL_END_WINDOW))
        self.assertEqual(strengths, sorted(strengths))
        for seed in SEEDS:
            scenario = gradual(self.df, seed=seed)
            means = [w[SHIFT_FEATURE].mean() for w in scenario.windows]
            self.assertLess(means[0], means[-1])
            self.assertGreater(psi(scenario.initial_train[SHIFT_FEATURE], scenario.windows[-1][SHIFT_FEATURE]),
                               PSI_ALERT)

    def test_all_scenarios_come_in_table_order(self):
        names = [s.name for s in all_scenarios(self.df, seed=42)]
        self.assertEqual(names, ['real_replay', 'stationary', 'sudden', 'gradual'])


if __name__ == '__main__':
    unittest.main()
