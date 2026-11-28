"""Person 2: classical model comparison using training/development records only.

No code in this module accesses the reserved final 20% of the dataset.
"""
import time
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.calibration import CalibratedClassifierCV

from src.features import add_pdays_features, build_preprocessor
from src.metrics import evaluate_binary
from src.settings import BASELINE_FEATURES, ECONOMIC_FEATURES, SEED

FEATURE_SETS = {'baseline_10': BASELINE_FEATURES,
                'baseline_plus_economic_15': BASELINE_FEATURES + ECONOMIC_FEATURES}


def preprocessor_for(feature_set):
    """Use the exact original baseline preprocessor; append scaled economic columns only in sensitivity arm."""
    if feature_set == 'baseline_10':
        original = build_preprocessor()
        original.named_steps['columns'].transformers[1][1].set_params(keep_empty_features=True)
        return original
    if feature_set != 'baseline_plus_economic_15':
        raise ValueError(feature_set)
    original = build_preprocessor()
    original.named_steps['columns'].transformers[1][1].set_params(keep_empty_features=True)
    transforms = list(original.named_steps['columns'].transformers)
    transforms.append(('economic', Pipeline([
        ('impute', SimpleImputer(strategy='median')),
        ('scale', StandardScaler())]), ECONOMIC_FEATURES))
    return Pipeline([
        ('pdays', FunctionTransformer(add_pdays_features)),
        ('columns', ColumnTransformer(transforms, remainder='drop'))])


def model_candidates():
    """Small, reproducible tuning grid. All methods are classical ML."""
    return {
        'Logistic Regression': [
            LogisticRegression(C=0.3, max_iter=2000, random_state=SEED),
            LogisticRegression(C=1.0, class_weight='balanced', max_iter=2000, random_state=SEED)],
        'Decision Tree': [
            DecisionTreeClassifier(max_depth=6, min_samples_leaf=40, random_state=SEED),
            DecisionTreeClassifier(max_depth=12, min_samples_leaf=20, class_weight='balanced', random_state=SEED)],
        'Random Forest': [
            RandomForestClassifier(n_estimators=100, max_depth=12, min_samples_leaf=8,
                                   n_jobs=-1, random_state=SEED),
            RandomForestClassifier(n_estimators=100, min_samples_leaf=5,
                                   class_weight='balanced_subsample', n_jobs=-1, random_state=SEED)],
        'Linear SVM': [
            LinearSVC(C=0.2, max_iter=6000, dual=False, random_state=SEED),
            LinearSVC(C=1.0, class_weight='balanced', max_iter=6000,
                      dual=False, random_state=SEED)]}


def positive_scores(model, X):
    """Returns probability scores for calibrated models, or SVM margin scores.
    Average precision and ROC-AUC accept both; never treat SVM margins as probabilities.
    """
    if hasattr(model, 'predict_proba'):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


def best_f1_threshold(y, scores):
    """Choose cutoff using development labels only; not an independent score estimate."""
    precision, recall, cutoffs = precision_recall_curve(y, scores)
    if len(cutoffs) == 0:
        raise ValueError('Need nonconstant classification scores')
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    idx = int(np.argmax(f1))
    return float(cutoffs[idx]), float(f1[idx])


def train_comparison(train, dev, verbose=True):
    """Choose configurations using the last 20% of TRAIN; tune threshold on DEV.

    Only accesses train/dev. No stream or held-out data is evaluated or selected on.
    """
    y_train = (train['y'] == 'yes').astype(int).to_numpy()
    y_dev = (dev['y'] == 'yes').astype(int).to_numpy()
    split = int(len(train) * 0.8)
    rows = []
    fitted = {}
    for feature_set, features in FEATURE_SETS.items():
        X_train = train[features].copy()
        X_dev = dev[features].copy()
        for model_name, candidates in model_candidates().items():
            best_idx, best_ap = None, -float('inf')
            for idx, estimator in enumerate(candidates):
                pipe = Pipeline([('prep', preprocessor_for(feature_set)),
                                 ('model', clone(estimator))])
                pipe.fit(X_train.iloc[:split], y_train[:split])
                scores = positive_scores(pipe, X_train.iloc[split:])
                ap = average_precision_score(y_train[split:], scores)
                if ap > best_ap:
                    best_idx, best_ap = idx, float(ap)
            best_estimator = clone(candidates[best_idx])
            pipe = Pipeline([('prep', preprocessor_for(feature_set)), ('model', best_estimator)])
            start = time.perf_counter()
            pipe.fit(X_train, y_train)
            train_seconds = time.perf_counter() - start
            scores = positive_scores(pipe, X_dev)
            cutoff, _ = best_f1_threshold(y_dev, scores)
            result = evaluate_binary(y_dev, scores, threshold=cutoff)
            result.update(feature_set=feature_set, model=model_name,
                          tuning_train_AP=best_ap, selected_config=best_idx,
                          training_seconds=round(train_seconds, 3),
                          score_type=('margin' if model_name == 'Linear SVM' else 'probability'),
                          dev_tuned=True)
            rows.append(result)
            fitted[(feature_set, model_name)] = pipe
            if verbose:
                print(f'{feature_set:25s} {model_name:20s} dev AP {result["pr_auc"]:.4f} '
                      f'F1 {result["f1"]:.3f} TP {result["tp"]} FP {result["fp"]}')
    summary = pd.DataFrame(rows).sort_values(['pr_auc', 'f1'], ascending=False).reset_index(drop=True)
    return summary, fitted


def shortlist(summary, n=2):
    """Top n distinct model families by dev AP; preferences remain provisional."""
    return summary.drop_duplicates('model').head(n).copy()


def top_features(pipe, feature_names, n=15):
    """Return linear coefficients or impurity importance for interpretation (not causality)."""
    model = pipe.named_steps['model']
    transformed_names = pipe.named_steps['prep'].named_steps['columns'].get_feature_names_out()
    if hasattr(model, 'coef_'):
        values = model.coef_[0]
    elif hasattr(model, 'feature_importances_'):
        values = model.feature_importances_
    else:
        return pd.DataFrame(columns=['feature', 'importance'])
    return (pd.DataFrame({'feature': transformed_names, 'importance': values})
            .assign(abs_importance=lambda d: d.importance.abs())
            .sort_values('abs_importance', ascending=False).head(n)
            .drop(columns='abs_importance').reset_index(drop=True))
