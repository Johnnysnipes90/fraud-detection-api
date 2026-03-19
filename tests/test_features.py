import pandas as pd

from src.features.engineer import (
    add_log_transaction_amount,
    add_missingness_indicators,
    engineer_features,
)


def test_add_log_transaction_amount():
    df = pd.DataFrame({"TransactionAmt": [100.0, 200.0]})
    result = add_log_transaction_amount(df)

    assert "log_TransactionAmt" in result.columns
    assert result["log_TransactionAmt"].notnull().all()


def test_add_missingness_indicators():
    df = pd.DataFrame(
        {
            "DeviceType": ["desktop", None],
            "dist1": [None, 2.0],
        }
    )
    result = add_missingness_indicators(df, columns=["DeviceType", "dist1"])

    assert "DeviceType_missing" in result.columns
    assert "dist1_missing" in result.columns
    assert result["DeviceType_missing"].tolist() == [0, 1]
    assert result["dist1_missing"].tolist() == [1, 0]


def test_engineer_features():
    df = pd.DataFrame(
        {
            "TransactionAmt": [50.0],
            "DeviceType": [None],
            "dist1": [1.0],
            "dist2": [None],
            "id_30": [None],
            "id_31": ["chrome"],
        }
    )
    result = engineer_features(df)

    assert "log_TransactionAmt" in result.columns
    assert "DeviceType_missing" in result.columns
    assert "dist2_missing" in result.columns
