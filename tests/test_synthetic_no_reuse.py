"""No scored row may ever be in any model's training history (synthetic scenarios, every seed).

Rows are identified by their original row number (the DataFrame index), which survives resampling, so a
resampled row can be traced back to the file row it came from.

Run from the project root:   python -m unittest discover -s tests -v
"""
import dataclasses
import sys
import unittest
import warnings
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import strategies
from src.data import load_dataset
from src.scenarios import gradual, real_replay, stationary, sudden
from src.settings import N_STREAM_WINDOWS, SEEDS

BUILDERS = {'stationary': stationary, 'sudden': sudden, 'gradual': gradual}
MODEL = 'Logistic Regression'


def scored_rows(scenario):
    """Original row numbers of every row that is ever scored."""
    return set().union(*[set(w.index) for w in scenario.windows])


class NoReuseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        warnings.simplefilter('ignore')
        cls.df = load_dataset()

    def test_no_history_ever_contains_a_scored_row(self):
        """Data level: initial rows and the labelled history before EVERY window, all scenarios and seeds."""
        for seed in SEEDS:
            for name, build in BUILDERS.items():
                scenario = build(self.df, seed=seed)
                scored = scored_rows(scenario)
                with self.subTest(scenario=name, seed=seed):
                    self.assertTrue(set(scenario.initial_train.index).isdisjoint(scored))
                    for i in range(N_STREAM_WINDOWS):
                        history = strategies.labelled_history(scenario, i)
                        self.assertTrue(set(history.index).isdisjoint(scored), f'window index {i}')
                    labelled = set().union(*[set(w.index) for w in scenario.labelled_windows])
                    self.assertTrue(labelled.isdisjoint(scored))

    def test_no_model_is_ever_fitted_on_a_scored_row(self):
        """Model level: spy on every fit made by the real strategy loop (initial fit and retrains)."""
        real_fit = strategies.fit_model
        for seed in SEEDS:
            for name, build in BUILDERS.items():
                scenario = build(self.df, seed=seed)
                scored = scored_rows(scenario)
                fits = []

                def spy(model_name, labelled_rows, *args, **kwargs):
                    fits.append(set(labelled_rows.index))
                    return real_fit(model_name, labelled_rows, *args, **kwargs)

                with mock.patch.object(strategies, 'fit_model', spy):
                    strategies.run_all_strategies(scenario, MODEL, seed=seed,
                                                  strategies=['periodic', 'drift_triggered'])
                with self.subTest(scenario=name, seed=seed):
                    self.assertGreater(len(fits), 2)       # the initial fit plus retrains really happened
                    for rows in fits:
                        self.assertTrue(rows.isdisjoint(scored))

    def test_the_check_would_catch_the_old_design(self):
        """Negative control: with labelled windows = the scored windows (the old behaviour) overlap appears."""
        scenario = dataclasses.replace(sudden(self.df, seed=42), labelled_windows=None)
        history = strategies.labelled_history(scenario, N_STREAM_WINDOWS - 1)
        self.assertFalse(set(history.index).isdisjoint(scored_rows(scenario)))

    def test_real_replay_history_holds_only_earlier_windows(self):
        scenario = real_replay(self.df)
        scored = scored_rows(scenario)
        self.assertTrue(set(scenario.initial_train.index).isdisjoint(scored))
        for i in range(N_STREAM_WINDOWS):
            self.assertTrue(set(strategies.labelled_history(scenario, i).index) - set(scenario.initial_train.index)
                            <= set().union(*[set(w.index) for w in scenario.windows[:max(0, i - 1)]]))


if __name__ == '__main__':
    unittest.main()
