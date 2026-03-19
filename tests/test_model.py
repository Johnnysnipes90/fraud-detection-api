import numpy as np

from src.models.evaluate import evaluate_predictions, threshold_sweep


def test_evaluate_predictions_returns_expected_keys():
    y_true = np.array([0, 1, 0, 1])
    y_prob = np.array([0.1, 0.9, 0.2, 0.8])

    metrics = evaluate_predictions(y_true, y_prob, threshold=0.5)

    expected_keys = {"roc_auc", "pr_auc", "precision", "recall", "f1"}
    assert expected_keys.issubset(metrics.keys())


def test_threshold_sweep_returns_dataframe():
    y_true = np.array([0, 1, 0, 1, 1])
    y_prob = np.array([0.05, 0.9, 0.2, 0.75, 0.6])

    result = threshold_sweep(y_true, y_prob, thresholds=[0.3, 0.5, 0.7])

    assert list(result.columns) == ["threshold", "precision", "recall", "f1"]
    assert len(result) == 3
