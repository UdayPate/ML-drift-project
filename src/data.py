"""Loading the dataset and splitting it chronologically."""
import pandas as pd

from src.settings import (DEV_FRAC, EXPECTED_COLUMNS, EXPECTED_ROWS, POSITIVE_LABEL, RAW_DATA_PATH,
                          TARGET, TRAIN_FRAC)


def load_dataset(path=RAW_DATA_PATH):
    """Read the full 41,188-row file in its original row order.

    `keep_default_na=False` makes sure the text "unknown" (and any other word) stays a plain string and is
    never silently turned into a missing value. The index is 0..n-1 and doubles as the row order.
    """
    df = pd.read_csv(path, sep=';', keep_default_na=False, na_values=[])
    if df.shape != (EXPECTED_ROWS, EXPECTED_COLUMNS):
        raise ValueError(f'Expected {(EXPECTED_ROWS, EXPECTED_COLUMNS)} (the full file), got {df.shape}')
    return df


def chronological_split(df):
    """Split into (train, development, stream) as contiguous blocks of the original row order.

    No shuffling: the first 60% of rows are train, the next 20% development, the last 20% stream.
    Each block keeps its original index labels, so row positions stay traceable.
    """
    n_rows = len(df)
    train_end = int(n_rows * TRAIN_FRAC)
    dev_end = int(n_rows * (TRAIN_FRAC + DEV_FRAC))
    return df.iloc[:train_end].copy(), df.iloc[train_end:dev_end].copy(), df.iloc[dev_end:].copy()


def encode_target(df):
    """1 for a subscription ("yes"), 0 otherwise."""
    return (df[TARGET] == POSITIVE_LABEL).astype(int)
