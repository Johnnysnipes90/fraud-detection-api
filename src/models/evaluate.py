from __future__ import annotations

from typing import Dict

import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_predictions(
    y_true,
    y_prob,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Evaluate predictions using fraud-appropriate metrics.
    """
    y_pred = (y_prob >= threshold).astype(int)

    return {
        "roc_auc": roc_auc_score(y_true, y_prob),
        "pr_auc": average_precision_score(y_true, y_prob),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }


def threshold_sweep(y_true, y_prob, thresholds=None) -> pd.DataFrame:
    """
    Evaluate precision/recall/F1 across thresholds.
    """
    if thresholds is None:
        thresholds = [round(x, 2) for x in [i / 100 for i in range(5, 95, 5)]]

    rows = []
    for t in thresholds:
        metrics = evaluate_predictions(y_true, y_prob, threshold=t)
        rows.append(
            {
                "threshold": t,
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
            }
        )

    return pd.DataFrame(rows)
