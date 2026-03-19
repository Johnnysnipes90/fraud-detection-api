from __future__ import annotations

from typing import Dict, List, Tuple

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.data.preprocess import (
    apply_frequency_encoding,
    build_preprocessor,
    split_categorical_by_cardinality,
    split_feature_types,
)
from src.features.engineer import engineer_features


def compute_scale_pos_weight(y: pd.Series) -> float:
    neg = (y == 0).sum()
    pos = (y == 1).sum()
    return float(neg / pos)


def build_xgb_pipeline(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> Tuple[Pipeline, pd.DataFrame, List[str], Dict[str, Dict]]:
    """
    Build fitted XGBoost pipeline and return artifacts needed for inference.
    """
    X_train = engineer_features(X_train)

    numeric_cols, categorical_cols = split_feature_types(X_train)
    low_card_cols, high_card_cols = split_categorical_by_cardinality(
        X_train, categorical_cols
    )

    X_train_fe, _, frequency_maps = apply_frequency_encoding(
        X_train, X_train.copy(), high_card_cols
    )

    numeric_cols_after = X_train_fe.select_dtypes(include=["number"]).columns.tolist()

    preprocessor = build_preprocessor(
        numeric_columns=numeric_cols_after,
        low_cardinality_columns=low_card_cols,
    )

    scale_pos_weight = compute_scale_pos_weight(y_train)

    model = XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="aucpr",
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1,
    )

    pipe = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )
    pipe.fit(X_train_fe, y_train)

    expected_columns = X_train_fe.columns.tolist()
    return pipe, X_train_fe, high_card_cols, frequency_maps


def save_training_artifacts(
    pipeline: Pipeline,
    expected_columns: List[str],
    high_card_cols: List[str],
    frequency_maps: Dict[str, Dict],
    model_path: str = "models/fraud_model.pkl",
    expected_columns_path: str = "models/expected_columns.pkl",
    high_card_cols_path: str = "models/high_card_cols.pkl",
    frequency_maps_path: str = "models/frequency_maps.pkl",
) -> None:
    joblib.dump(pipeline, model_path)
    joblib.dump(expected_columns, expected_columns_path)
    joblib.dump(high_card_cols, high_card_cols_path)
    joblib.dump(frequency_maps, frequency_maps_path)
