import unittest
import numpy as np
from src.model_comparison import best_f1_threshold, FEATURE_SETS, preprocessor_for
from src.settings import ALWAYS_EXCLUDED

class Person2Checks(unittest.TestCase):
    def test_features_do_not_include_duration(self):
        for values in FEATURE_SETS.values():
            self.assertFalse(set(values) & set(ALWAYS_EXCLUDED))
    def test_threshold_uses_positive_scores(self):
        y = np.array([0, 0, 1, 1])
        scores = np.array([0.1, 0.2, 0.8, 0.9])
        cutoff, best = best_f1_threshold(y, scores)
        self.assertAlmostEqual(best, 1.0)
        self.assertGreater(cutoff, 0.2)
    def test_both_preprocessors_build(self):
        for group in FEATURE_SETS:
            self.assertIsNotNone(preprocessor_for(group))

if __name__ == '__main__':
    unittest.main()
