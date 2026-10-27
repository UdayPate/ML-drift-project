"""Phase A checks: loading, chronological split, feature selection and leakage-safe preprocessing.

Run from the project root:   python -m unittest discover -s tests -v
Expected values are written out literally on purpose, so a wrong setting in `src/settings.py` is caught.
"""
import csv
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # project root, same idea as in notebooks

from src.data import chronological_split, encode_target, load_dataset
from src.features import (CATEGORICAL_COLUMNS, add_pdays_features, build_preprocessor,
                          select_baseline_features)
from src.settings import RAW_DATA_PATH

EXPECTED_TEN_FEATURES = ['age', 'job', 'marital', 'education', 'default', 'housing', 'loan',
                         'pdays', 'previous', 'poutcome']
TRAIN_END, DEV_END, N_ROWS = 24_712, 32_950, 41_188


def fitted_parameters(preprocessor):
    """Everything the preprocessing pipeline learned during fit, as plain comparable objects."""
    columns = preprocessor.named_steps['columns']
    scaler = columns.named_transformers_['scaled']
    imputer = columns.named_transformers_['imputed']
    encoder = columns.named_transformers_['categorical']
    return {
        'impute_statistics': imputer.statistics_,          # [median of pdays_days]
        'scale_mean': scaler.mean_,                        # [mean of age]
        'scale_std': scaler.scale_,                        # [spread of age]
        'categories': {c: list(cats) for c, cats in zip(CATEGORICAL_COLUMNS, encoder.categories_)},
    }


def assert_same_parameters(test, a, b):
    np.testing.assert_array_equal(a['impute_statistics'], b['impute_statistics'])
    np.testing.assert_array_equal(a['scale_mean'], b['scale_mean'])
    np.testing.assert_array_equal(a['scale_std'], b['scale_std'])
    test.assertEqual(a['categories'], b['categories'])


def to_dense(matrix):
    return matrix.toarray() if sparse.issparse(matrix) else np.asarray(matrix)


class PhaseATests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = load_dataset()
        cls.train, cls.dev, cls.stream = chronological_split(cls.df)

    # 1 ---------------------------------------------------------------------------------------------
    def test_full_file_loaded_in_original_row_order(self):
        with open(RAW_DATA_PATH, newline='', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter=';')
            header = next(reader)
            raw_rows = list(reader)
        self.assertEqual(len(raw_rows), N_ROWS)
        self.assertEqual(self.df.shape, (N_ROWS, 21))
        self.assertTrue(self.df.index.equals(pd.RangeIndex(N_ROWS)))
        for column in ['age', 'job', 'marital', 'education', 'campaign', 'pdays', 'previous', 'y']:
            raw = [row[header.index(column)] for row in raw_rows]
            self.assertEqual(self.df[column].astype(str).tolist(), raw, f'row order differs in {column}')
        raw_euribor = [float(row[header.index('euribor3m')]) for row in raw_rows]
        np.testing.assert_allclose(self.df['euribor3m'].to_numpy(), raw_euribor)
        self.assertEqual(int((self.df['y'] == 'yes').sum()), 4_640)

    def test_unknown_is_kept_as_text_not_missing(self):
        self.assertEqual(int(self.df.isna().sum().sum()), 0)
        self.assertEqual(int((self.df['default'] == 'unknown').sum()), 8_597)

    # 2 ---------------------------------------------------------------------------------------------
    def test_split_boundaries_and_no_shuffling(self):
        self.assertEqual((len(self.train), len(self.dev), len(self.stream)), (24_712, 8_238, 8_238))
        self.assertEqual((self.train.index[0], self.train.index[-1]), (0, TRAIN_END - 1))
        self.assertEqual((self.dev.index[0], self.dev.index[-1]), (TRAIN_END, DEV_END - 1))
        self.assertEqual((self.stream.index[0], self.stream.index[-1]), (DEV_END, N_ROWS - 1))
        for part in (self.train, self.dev, self.stream):
            self.assertTrue(part.index.is_monotonic_increasing)
            self.assertEqual(part.index.tolist(), list(range(part.index[0], part.index[-1] + 1)))
        pd.testing.assert_frame_equal(pd.concat([self.train, self.dev, self.stream]), self.df)

    def test_split_class_counts_match_the_feasibility_inspection(self):
        self.assertEqual((int(encode_target(self.train).sum()), len(self.train)), (1_188, 24_712))
        self.assertEqual((int(encode_target(self.dev).sum()), len(self.dev)), (912, 8_238))

    # 3 ---------------------------------------------------------------------------------------------
    def test_exactly_the_ten_baseline_features_are_selected(self):
        selected = select_baseline_features(self.df)
        self.assertEqual(list(selected.columns), EXPECTED_TEN_FEATURES)
        not_allowed = ['duration', 'y', 'campaign', 'contact', 'month', 'day_of_week',
                       'emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed']
        for column in not_allowed:
            self.assertNotIn(column, selected.columns)

    # 4 ---------------------------------------------------------------------------------------------
    def test_pdays_999_is_a_sentinel_and_no_rows_are_dropped(self):
        X = select_baseline_features(self.df)
        out = add_pdays_features(X)
        self.assertEqual(len(out), N_ROWS)
        self.assertNotIn('pdays', out.columns)
        is_999 = X['pdays'] == 999
        np.testing.assert_array_equal(out['pdays_is_999'].to_numpy(), is_999.astype(int).to_numpy())
        self.assertTrue(out.loc[is_999, 'pdays_days'].isna().all())
        self.assertTrue(out.loc[~is_999, 'pdays_days'].notna().all())
        self.assertFalse((out['pdays_days'] == 999).any())
        np.testing.assert_array_equal(out.loc[~is_999, 'pdays_days'].to_numpy(), X.loc[~is_999, 'pdays'].to_numpy())
        inconsistent = (X['pdays'] == 999) & (X['previous'] > 0)       # documented interpretation issue
        self.assertEqual(int(inconsistent.sum()), 4_110)
        self.assertEqual(int(out.loc[inconsistent, 'pdays_is_999'].sum()), 4_110)   # kept, flagged as 999

    def test_pipeline_output_has_no_missing_values_and_no_raw_999(self):
        X_train = select_baseline_features(self.train)
        prep = build_preprocessor().fit(X_train)
        for part in (self.train, self.dev, self.stream):
            matrix = to_dense(prep.transform(select_baseline_features(part)))
            self.assertEqual(matrix.shape[0], len(part))
            self.assertFalse(np.isnan(matrix).any())
        n_categories = sum(len(c) for c in fitted_parameters(prep)['categories'].values())
        width = 4 + n_categories    # age, pdays_days, previous, pdays_is_999, then the one-hot columns
        self.assertEqual(to_dense(prep.transform(X_train)).shape, (len(X_train), width))

    # 5 ---------------------------------------------------------------------------------------------
    def test_unknown_is_learned_as_its_own_category(self):
        prep = build_preprocessor().fit(select_baseline_features(self.train))
        categories = fitted_parameters(prep)['categories']
        for column in CATEGORICAL_COLUMNS:
            if (self.train[column] == 'unknown').any():
                self.assertIn('unknown', categories[column], column)
        self.assertEqual(sorted(categories['default']), ['no', 'unknown', 'yes'])

    # 6 ---------------------------------------------------------------------------------------------
    def test_preprocessing_statistics_come_from_training_rows_only(self):
        X_train = select_baseline_features(self.train)
        prep = build_preprocessor().fit(X_train)
        params = fitted_parameters(prep)
        np.testing.assert_allclose(params['scale_mean'][0], X_train['age'].mean())
        train_pdays_median = X_train.loc[X_train['pdays'] != 999, 'pdays'].median()
        self.assertEqual(params['impute_statistics'][0], train_pdays_median)
        self.assertEqual(train_pdays_median, 4.0)                                    # from only 6 real values
        self.assertNotAlmostEqual(params['scale_mean'][0], self.df['age'].mean(), places=6)

    def test_transforming_other_rows_does_not_change_what_was_learned(self):
        prep = build_preprocessor().fit(select_baseline_features(self.train))
        before = fitted_parameters(prep)
        for part in (self.dev, self.stream):
            prep.transform(select_baseline_features(part))
        assert_same_parameters(self, before, fitted_parameters(prep))

    # 6b --------------------------------------------------------------------------------------------
    def test_previous_and_pdays_days_are_not_standardised_but_age_is(self):
        """Output columns are: age, pdays_days, previous, pdays_is_999, then the one-hot columns."""
        X_train = select_baseline_features(self.train)
        prep = build_preprocessor().fit(X_train)
        params = fitted_parameters(prep)
        self.assertEqual(int((X_train['pdays'] != 999).sum()), 6)          # the reason for this choice
        self.assertEqual(int(X_train['previous'].max()), 1)
        for part in (self.train, self.dev):
            X = select_baseline_features(part)
            out = to_dense(prep.transform(X))
            np.testing.assert_array_equal(out[:, 2], X['previous'].to_numpy())          # previous: raw
            np.testing.assert_array_equal(out[:, 3], (X['pdays'] == 999).astype(int).to_numpy())
            real = (X['pdays'] != 999).to_numpy()
            np.testing.assert_array_equal(out[real, 1], X.loc[real, 'pdays'].to_numpy())  # real days: raw
            self.assertTrue((out[~real, 1] == params['impute_statistics'][0]).all())      # 999 rows: median
            self.assertLessEqual(float(out[:, 1].max()), 27.0)                            # no extreme values
            np.testing.assert_allclose(out[:, 0], (X['age'] - params['scale_mean'][0]) / params['scale_std'][0])
        self.assertEqual(len(params['scale_mean']), 1)                    # only age has a scaler

    # 7 / 9 -----------------------------------------------------------------------------------------
    def test_changing_development_and_stream_rows_does_not_change_the_fitted_preprocessing(self):
        """Corrupt every feature and label after the training rows, redo split + fit, compare."""
        corrupted = self.df.copy()
        corrupted.loc[TRAIN_END:, ['age', 'previous', 'pdays']] = [99, 7, 3]
        for column in CATEGORICAL_COLUMNS:
            corrupted.loc[TRAIN_END:, column] = 'never-seen-in-training'
        corrupted.loc[TRAIN_END:, 'y'] = 'yes'

        train_a, _, _ = chronological_split(self.df)
        train_b, dev_b, stream_b = chronological_split(corrupted)
        pd.testing.assert_frame_equal(train_a, train_b)          # training rows really are untouched
        self.assertTrue((dev_b['age'] == 99).all() and (stream_b['job'] == 'never-seen-in-training').all())

        prep_a = build_preprocessor().fit(select_baseline_features(train_a))
        prep_b = build_preprocessor().fit(select_baseline_features(train_b))     # fit receives no labels
        assert_same_parameters(self, fitted_parameters(prep_a), fitted_parameters(prep_b))
        np.testing.assert_array_equal(to_dense(prep_a.transform(select_baseline_features(train_a))),
                                      to_dense(prep_b.transform(select_baseline_features(train_b))))
        # Unseen categories in later data must be encoded as zeros, not crash and not extend the categories.
        to_dense(prep_b.transform(select_baseline_features(dev_b)))
        self.assertNotIn('never-seen-in-training', fitted_parameters(prep_b)['categories']['job'])


if __name__ == '__main__':
    unittest.main()
