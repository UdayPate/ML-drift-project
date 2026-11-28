"""Checks for src/drift.py: PSI and KS behave as expected, and the trigger rule fires only on real change.

Run from the project root:   python -m unittest discover -s tests -v
Small, fast synthetic data is used except where the real file is needed.
"""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.drift import (drift_over_windows, drift_report, drift_trigger, false_alarm_rate, is_categorical, ks,
                       null_psi_distribution, psi)
from src.settings import PSI_ALERT


class PsiTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(0)

    def test_identical_samples_give_zero(self):
        x = pd.Series(self.rng.normal(size=2_000))
        self.assertAlmostEqual(psi(x, x), 0.0, places=12)

    def test_same_distribution_is_small_and_shift_is_large(self):
        reference = pd.Series(self.rng.normal(size=5_000))
        same = pd.Series(self.rng.normal(size=1_000))
        shifted = pd.Series(self.rng.normal(loc=1.0, size=1_000))
        self.assertLess(psi(reference, same), 0.1)
        self.assertGreater(psi(reference, shifted), PSI_ALERT)

    def test_psi_is_never_negative(self):
        reference = pd.Series(self.rng.integers(0, 50, size=1_000))
        for _ in range(20):
            current = pd.Series(self.rng.integers(0, 60, size=300))
            self.assertGreaterEqual(psi(reference, current), 0.0)

    def test_categorical_and_new_category(self):
        reference = pd.Series(['a'] * 80 + ['b'] * 20)
        self.assertTrue(is_categorical(reference))
        self.assertAlmostEqual(psi(reference, reference.copy()), 0.0, places=12)
        # A category never seen in the reference must not crash and must count as drift.
        current = pd.Series(['a'] * 50 + ['c'] * 50)
        self.assertGreater(psi(reference, current), PSI_ALERT)

    def test_few_valued_numeric_is_treated_as_categories(self):
        self.assertTrue(is_categorical(pd.Series([0, 1, 0, 0, 1, 2])))
        self.assertFalse(is_categorical(pd.Series(np.arange(100, dtype=float))))

    def test_current_values_outside_reference_range_are_counted(self):
        reference = pd.Series(np.arange(1_000, dtype=float))
        current = pd.Series(np.full(200, 5_000.0))          # all far above the reference maximum
        self.assertGreater(psi(reference, current), PSI_ALERT)

    def test_empty_sample_is_refused(self):
        with self.assertRaises(ValueError):
            psi(pd.Series([1.0, 2.0]), pd.Series([], dtype=float))


class KsTests(unittest.TestCase):
    def test_ks_numeric_and_text(self):
        rng = np.random.default_rng(1)
        stat, p = ks(pd.Series(rng.normal(size=500)), pd.Series(rng.normal(loc=2, size=500)))
        self.assertGreater(stat, 0.5)
        self.assertLess(p, 0.001)
        stat, p = ks(pd.Series(['a', 'b']), pd.Series(['a', 'a']))
        self.assertTrue(np.isnan(stat) and np.isnan(p))


class TriggerTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(2)
        self.groups = {'model': ['x', 'kind'], 'economy': ['rate']}
        self.reference = pd.DataFrame({'x': rng.normal(size=4_000),
                                       'kind': rng.choice(['p', 'q'], size=4_000),
                                       'rate': rng.normal(size=4_000)})
        self.same = self.reference.sample(1_000, random_state=3)
        self.economy_only = self.same.assign(rate=self.same['rate'] + 3)
        self.model_shift = self.same.assign(x=self.same['x'] + 1.5)

    def test_report_has_one_row_per_feature(self):
        report = drift_report(self.reference, self.same, groups=self.groups)
        self.assertEqual(sorted(report['feature']), ['kind', 'rate', 'x'])
        self.assertEqual(set(report.columns), {'group', 'feature', 'psi', 'ks_stat', 'ks_p', 'alert'})

    def test_no_change_does_not_trigger(self):
        decision = drift_trigger(self.reference, self.same, trigger_groups=['model'], groups=self.groups)
        self.assertFalse(decision['retrain'])
        self.assertEqual(decision['fired'], [])

    def test_shift_in_trigger_group_triggers(self):
        decision = drift_trigger(self.reference, self.model_shift, trigger_groups=['model'], groups=self.groups)
        self.assertTrue(decision['retrain'])
        self.assertEqual(decision['fired'], ['x'])

    def test_shift_outside_trigger_group_is_logged_but_does_not_trigger(self):
        decision = drift_trigger(self.reference, self.economy_only, trigger_groups=['model'], groups=self.groups)
        self.assertFalse(decision['retrain'])
        economy = decision['report'].set_index('feature').loc['rate']
        self.assertTrue(economy['alert'])          # still measured and flagged in the report

    def test_drift_over_windows_numbers_windows_from_one(self):
        result = drift_over_windows(self.reference, [self.same, self.model_shift], groups=self.groups)
        self.assertEqual(sorted(result['window'].unique()), [1, 2])

    def test_null_distribution_gives_low_false_alarm_rate(self):
        null = null_psi_distribution(self.reference, window_rows=1_000, n_draws=20, groups=self.groups)
        self.assertEqual(len(null), 20 * 3)
        self.assertEqual(false_alarm_rate(null, trigger_groups=['model']), 0.0)


if __name__ == '__main__':
    unittest.main()
