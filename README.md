# 🚀 Fraud Detection System (End-to-End ML + Live API)

A production-style fraud detection system built using machine learning, designed to identify high-risk transactions with tunable precision-recall tradeoffs.

This project covers the full ML lifecycle:

- Data exploration
- Feature engineering
- Model training
- Time-based validation
- Threshold optimization
- Explainability (SHAP)
- API deployment with FastAPI

---

## 🌐 Live API

Base URL:

```
https://fraud-detection-api-dohb.onrender.com

```

### Interactive Docs

```
https://fraud-detection-api-dohb.onrender.com/docs

```

### Health Check

```
https://fraud-detection-api-dohb.onrender.com/health

```

---

## 📌 Problem

Fraud detection is a highly imbalanced classification problem where:

- Fraud cases are rare (~3.5%)
- Missing fraud is costly
- False positives impact customer experience

The goal is to build a model that can:

- Detect fraudulent transactions effectively
- Provide controllable tradeoffs between recall and precision
- Operate realistically in a production setting

---

## 🧠 Approach

### 1. Data Processing

- Merged transaction + identity datasets
- Handled high missingness (214+ columns >50% missing)
- Created missingness indicators (important fraud signals)

### 2. Feature Engineering

- Log transformation of `TransactionAmt`
- Frequency encoding for high-cardinality features
- One-hot encoding for low-cardinality features

### 3. Modeling

Models trained:

- Logistic Regression (baseline)
- Random Forest
- **XGBoost (final model)**

### 4. Validation Strategy

- Random split (baseline)
- **Time-based split (realistic)**
- **Rolling time validation (production-level robustness)**

### 5. Imbalance Handling

- `scale_pos_weight` for XGBoost
- Focus on PR-AUC, Recall, Precision, F1

---

## 📊 Key Results

### Model Performance (XGBoost)

| Validation   | ROC-AUC | PR-AUC | Precision | Recall |     F1 |
| ------------ | ------: | -----: | --------: | -----: | -----: |
| Random Split |  0.9388 | 0.6612 |    0.2486 | 0.8193 | 0.3814 |
| Time-Based   |  0.9051 | 0.5090 |    0.2059 | 0.7431 | 0.3225 |

---

## 🎯 Final Decision Threshold

Threshold: **0.85**

- Precision: **0.60**
- Recall: **0.42**
- F1: **0.49**

✔ High-confidence fraud detection  
✔ Reduced false positives  
✔ Business-ready deployment

---

## ⚖️ Business Tradeoff

| Threshold | Precision | Recall | Use Case                    |
| --------- | --------- | ------ | --------------------------- |
| 0.50      | Low       | High   | Aggressive fraud detection  |
| 0.85      | High      | Medium | Balanced / production-ready |

---

## ⚙️ API Usage

### Endpoint

**POST/predict**

```
{
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
  "id_35": "T"
}
```

### Example Response

```
{
  "fraud_probability": 0.667384,
  "fraud_prediction": 0,
  "threshold_used": 0.85,
  "decision": "approve"
}
```

---

## 🔍 Explainability (SHAP)

- Global feature importance
- Individual transaction explanations
- Fraud driver analysis
  This ensures:

- transparency
- auditability
- trust in model decisions

---

## 🧪 Testing

```
pytest

```

---

## 🐳 Docker

```
docker build -t fraud-detection-api .
docker run -p 8000:8000 fraud-detection-api
```

---

## 👤 Author

John Olalemi Data Scientist | Product Analytics | Machine Learning

🔗 GitHub: https://github.com/Johnnysnipes90
