"""Checks for src/metrics.py against values worked out by hand.

Run from the project root:   python -m unittest discover -s tests -v
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.metrics import evaluate_binary

# Six rows. Sorted by score (high to low): 0.8(yes) 0.7(yes) 0.6(no) 0.4(no) 0.35(yes) 0.1(no).
Y_TRUE = [0, 0, 1, 1, 1, 0]
SCORES = [0.1, 0.4, 0.35, 0.8, 0.7, 0.6]


class MetricsTests(unittest.TestCase):
    def test_values_worked_out_by_hand(self):
        m = evaluate_binary(Y_TRUE, SCORES, threshold=0.5)
        # Predicted positive at 0.5: the rows scored 0.8, 0.7, 0.6 -> two real yes, one real no.
        self.assertEqual((m['tn'], m['fp'], m['fn'], m['tp']), (2, 1, 1, 2))
        self.assertAlmostEqual(m['precision'], 2 / 3)
        self.assertAlmostEqual(m['recall'], 2 / 3)
        self.assertAlmostEqual(m['f1'], 2 / 3)
        self.assertEqual((m['n_predicted_positive'], m['pct_predicted_positive']), (3, 0.5))
        self.assertAlmostEqual(m['prevalence'], 0.5)
        self.assertAlmostEqual(m['random_pr_auc'], 0.5)              # random ranking = prevalence
        # ROC-AUC: 7 of the 9 (yes, no) pairs are ranked correctly.
        self.assertAlmostEqual(m['roc_auc'], 7 / 9)
        # PR-AUC (average precision): precision at each yes is 1/1, 2/2, 3/5 -> (1 + 1 + 0.6) / 3.
        self.assertAlmostEqual(m['pr_auc'], 2.6 / 3)

    def test_a_score_exactly_at_the_threshold_is_predicted_positive(self):
        m = evaluate_binary([1, 0], [0.5, 0.49], threshold=0.5)
        self.assertEqual((m['tp'], m['fp'], m['fn'], m['tn']), (1, 0, 0, 1))

    def test_nothing_predicted_positive_gives_zero_precision_not_an_error(self):
        m = evaluate_binary([1, 0, 0], [0.2, 0.1, 0.3], threshold=0.5)
        self.assertEqual((m['tp'], m['fp'], m['fn'], m['tn']), (0, 0, 1, 2))
        self.assertEqual((m['precision'], m['recall'], m['f1'], m['n_predicted_positive']), (0.0, 0.0, 0.0, 0))

    def test_default_threshold_comes_from_settings(self):
        from src.settings import DECISION_THRESHOLD
        self.assertEqual(DECISION_THRESHOLD, 0.5)
        self.assertEqual(evaluate_binary(Y_TRUE, SCORES)['threshold'], 0.5)

    def test_random_ranking_reference_equals_prevalence(self):
        m = evaluate_binary([1, 0, 0, 0], [0.9, 0.1, 0.2, 0.3])
        self.assertAlmostEqual(m['random_pr_auc'], 0.25)


if __name__ == '__main__':
    unittest.main()
