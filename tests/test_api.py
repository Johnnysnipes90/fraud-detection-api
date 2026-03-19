from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_root():
    response = client.get("/")
    assert response.status_code == 200
    body = response.json()
    assert "message" in body
    assert "threshold" in body


def test_predict():
    payload = {
        "TransactionDT": 12000000,
        "TransactionAmt": 350.0,
        "ProductCD": "C",
        "card4": "visa",
        "card6": "credit",
        "P_emaildomain": "gmail.com",
        "R_emaildomain": "gmail.com",
        "DeviceType": "desktop",
        "DeviceInfo": "Windows",
        "dist1": 12.0,
        "dist2": 50.0,
        "C1": 4.0,
        "C4": 2.0,
        "C8": 1.0,
        "C10": 0.0,
        "C14": 1.0,
        "D1": 5.0,
        "D4": 2.0,
        "id_30": "Windows 10",
        "id_31": "chrome 120.0",
        "id_35": "T",
    }

    response = client.post("/predict", json=payload)
    assert response.status_code == 200, response.text

    body = response.json()
    assert "fraud_probability" in body
    assert "fraud_prediction" in body
    assert "threshold_used" in body
    assert "decision" in body
