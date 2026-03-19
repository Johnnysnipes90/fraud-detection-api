from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

DEFAULT_MISSING_SIGNAL_COLS = ["DeviceType", "dist1", "dist2", "id_30", "id_31"]


def add_log_transaction_amount(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add log-transformed transaction amount.
    """
    df = df.copy()
    if "TransactionAmt" in df.columns:
        df["log_TransactionAmt"] = np.log1p(df["TransactionAmt"])
    return df


def add_missingness_indicators(
    df: pd.DataFrame,
    columns: Iterable[str] = DEFAULT_MISSING_SIGNAL_COLS,
) -> pd.DataFrame:
    """
    Add missing-value indicator columns for selected variables.
    """
    df = df.copy()
    for col in columns:
        if col in df.columns:
            df[f"{col}_missing"] = df[col].isnull().astype(int)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all feature engineering steps used in notebooks/API.
    """
    df = add_log_transaction_amount(df)
    df = add_missingness_indicators(df)
    return df
