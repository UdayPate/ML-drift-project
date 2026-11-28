"""The optional `scores_out` argument of `run_strategy` records per-row scores and changes nothing else.

Uses the small fake stream from test_strategies.py, so it runs without the dataset file.
Run from the project root:   python -m unittest discover -s tests -v
"""
import sys
import unittest
import warnings
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.strategies import run_strategy
from test_strategies import MODEL, fake_scenario

TIMING = ['retrain_seconds', 'monitor_seconds']


class ScoresOutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        warnings.simplefilter('ignore')
        cls.scenario = fake_scenario(shift_from=3)

    def test_recording_scores_does_not_change_the_results(self):
        for strategy in ('static', 'periodic', 'drift_triggered'):
            plain = run_strategy(self.scenario, strategy, MODEL, seed=42)
            out = []
            recorded = run_strategy(self.scenario, strategy, MODEL, seed=42, scores_out=out)
            pd.testing.assert_frame_equal(plain.drop(columns=TIMING), recorded.drop(columns=TIMING))

    def test_one_entry_per_window_with_matching_rows_labels_and_scores(self):
        out = []
        run_strategy(self.scenario, 'periodic', MODEL, seed=42, scores_out=out)
        self.assertEqual([e['window'] for e in out], list(range(1, len(self.scenario.windows) + 1)))
        for entry, window in zip(out, self.scenario.windows):
            self.assertEqual(entry['strategy'], 'periodic')
            self.assertEqual(list(entry['index']), list(window.index))
            self.assertEqual(len(entry['y']), len(window))
            self.assertEqual(len(entry['scores']), len(window))
            self.assertEqual(set(entry['y']), {0, 1})


if __name__ == '__main__':
    unittest.main()
