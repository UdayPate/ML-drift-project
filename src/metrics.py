"""Evaluation metrics for a binary classifier whose positive class is subscription = yes."""
import numpy as np
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)

from src.settings import DECISION_THRESHOLD


def evaluate_binary(y_true, scores, threshold=DECISION_THRESHOLD):
    """Compute ranking metrics and threshold-based metrics for predicted probabilities.

    y_true: 0/1 labels (1 = subscribed). scores: predicted probability of the positive class.

    Ranking metrics (do not depend on the threshold):
      - pr_auc: area under the precision-recall curve (`average_precision_score`), the primary metric
      - random_pr_auc: PR-AUC of random ranking, which equals the positive rate (prevalence)
      - roc_auc
    Threshold metrics (a row is predicted positive when score >= threshold):
      - precision, recall, f1, the confusion counts, and how many rows were predicted positive.
    Precision is reported as 0.0 when nothing is predicted positive (scikit-learn's `zero_division=0`).
    """
    y_true = np.asarray(y_true)
    scores = np.asarray(scores, dtype=float)
    predicted = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predicted, labels=[0, 1]).ravel()
    n_rows = len(y_true)
    return {
        'n_rows': n_rows,
        'n_positive': int(y_true.sum()),
        'prevalence': float(y_true.mean()),
        'pr_auc': float(average_precision_score(y_true, scores)),
        'random_pr_auc': float(y_true.mean()),
        'roc_auc': float(roc_auc_score(y_true, scores)),
        'threshold': float(threshold),
        'precision': float(precision_score(y_true, predicted, zero_division=0)),
        'recall': float(recall_score(y_true, predicted, zero_division=0)),
        'f1': float(f1_score(y_true, predicted, zero_division=0)),
        'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp),
        'n_predicted_positive': int(predicted.sum()),
        'pct_predicted_positive': float(predicted.mean()),
    }
