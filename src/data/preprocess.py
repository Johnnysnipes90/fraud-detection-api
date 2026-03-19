from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def drop_extremely_sparse_columns(
    df: pd.DataFrame,
    threshold: float = 0.95,
    exclude: List[str] | None = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Drop columns whose missing ratio exceeds the threshold.
    """
    exclude = exclude or []
    missing_ratio = df.isnull().mean()
    cols_to_drop = [
        col
        for col in missing_ratio[missing_ratio > threshold].index.tolist()
        if col not in exclude
    ]
    return df.drop(columns=cols_to_drop), cols_to_drop


def split_feature_types(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """
    Return numeric and categorical columns.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    return numeric_cols, categorical_cols


def split_categorical_by_cardinality(
    df: pd.DataFrame,
    categorical_cols: List[str],
    low_card_threshold: int = 10,
) -> Tuple[List[str], List[str]]:
    """
    Split categorical columns into low- and high-cardinality groups.
    """
    low_card_cols = [
        c for c in categorical_cols if df[c].nunique(dropna=True) <= low_card_threshold
    ]
    high_card_cols = [
        c for c in categorical_cols if df[c].nunique(dropna=True) > low_card_threshold
    ]
    return low_card_cols, high_card_cols


def build_frequency_maps(df: pd.DataFrame, columns: List[str]) -> Dict[str, Dict]:
    """
    Build frequency-encoding maps from training data only.
    """
    frequency_maps: Dict[str, Dict] = {}
    for col in columns:
        frequency_maps[col] = (
            df[col].value_counts(dropna=False, normalize=True).to_dict()
        )
    return frequency_maps


def apply_frequency_encoding(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    columns: List[str],
    frequency_maps: Dict[str, Dict] | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Dict]]:
    """
    Frequency-encode high-cardinality columns.
    """
    train_df = train_df.copy()
    valid_df = valid_df.copy()

    if frequency_maps is None:
        frequency_maps = build_frequency_maps(train_df, columns)

    for col in columns:
        freq_map = frequency_maps.get(col, {})
        train_df[col] = train_df[col].map(freq_map).fillna(0.0)
        valid_df[col] = valid_df[col].map(freq_map).fillna(0.0)

    return train_df, valid_df, frequency_maps


def build_preprocessor(
    numeric_columns: List[str],
    low_cardinality_columns: List[str],
) -> ColumnTransformer:
    """
    Build sklearn preprocessing pipeline.
    """
    numeric_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median"))]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_columns),
            ("cat", categorical_transformer, low_cardinality_columns),
        ]
    )
