# Customer Churn Prediction — End-to-End ML System

[![CI](https://github.com/PrathamPatil17/churn-prediction/actions/workflows/ci.yml/badge.svg)](https://github.com/PrathamPatil17/churn-prediction/actions)
[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://python.org)
[![MLflow](https://img.shields.io/badge/MLflow-2.9-orange)](https://mlflow.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.29-red)](https://streamlit.io)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0-green)](https://xgboost.readthedocs.io)

> Predicts which telecom customers will churn using an end-to-end ML pipeline —
> from raw data to a deployed, explainable model with a live business impact dashboard.

---

## Business Problem

A telecom company loses **26.5% of customers annually** to churn. At ~$1,200 average
annual value per customer, identifying and intervening with at-risk customers before
they leave is worth millions in recovered revenue. This system predicts churn probability
for each customer and explains the drivers behind every prediction.

---

## Results

| Model | ROC-AUC | F1 | Precision | Recall | PR-AUC |
|---|---|---|---|---|---|
| **XGBoost** | **0.912** | **0.634** | **0.709** | **0.574** | **0.718** |
| LightGBM | 0.907 | 0.621 | 0.698 | 0.561 | 0.701 |
| Random Forest | 0.889 | 0.598 | 0.681 | 0.534 | 0.672 |
| Logistic Regression | 0.841 | 0.571 | 0.634 | 0.521 | 0.631 |

At the **optimal decision threshold (0.38)**, the XGBoost model recovers an estimated
**$142,000 in net revenue** on a 1,400-customer test set, assuming $1,200 annual value
and $50 intervention cost per customer.

---

## Key Findings (EDA + SHAP)

1. **Contract type** is the single strongest predictor — month-to-month customers
   churn at 3× the rate of two-year contract customers
2. **Tenure** has the largest negative SHAP values — every additional month with the
   company substantially reduces churn probability
3. **Fiber optic customers** churn more despite paying more, suggesting a value
   perception gap that the product team should address
4. **No tech support + high monthly charges + short tenure** = the highest-risk
   customer segment (intervention priority tier 1)
5. **SMOTE** improved recall on the minority class from 0.41 → 0.57 without
   sacrificing precision meaningfully

---

## Architecture

```
data/                          Raw CSV (IBM Telco, 7,043 customers)
src/
├── data_loader.py             Load → clean → encode → feature engineer
├── train.py                   Train 4 models with MLflow tracking + SMOTE
├── evaluate.py                ROC/PR curves, confusion matrix, business impact
└── explain.py                 SHAP global + local explainability
dashboard/
└── app.py                     5-page Streamlit dashboard
notebooks/
├── 01_eda.py                  EDA and business insights
├── 02_train_and_evaluate.py   Full training run
└── 03_shap_analysis.py        Deep SHAP explainability
tests/
└── test_pipeline.py           Unit tests (pytest)
.github/workflows/ci.yml       CI pipeline with model quality gate (AUC ≥ 0.84)
```

---

## Quick Start

```bash
# 1. Clone and install
git clone https://github.com/PrathamPatil17/churn-prediction
cd churn-prediction
pip install -r requirements.txt

# 2. Download dataset
# From: https://www.kaggle.com/datasets/blastchar/telco-customer-churn
# Place CSV at: data/WA_Fn-UseC_-Telco-Customer-Churn.csv

# 3. Run EDA
python notebooks/01_eda.py

# 4. Train all models (tracked in MLflow)
python notebooks/02_train_and_evaluate.py

# 5. Open MLflow UI (in a separate terminal)
mlflow ui --port 5000
# Visit: http://localhost:5000

# 6. SHAP explainability analysis
python notebooks/03_shap_analysis.py

# 7. Launch dashboard
streamlit run dashboard/app.py
# Visit: http://localhost:8501

# 8. Run tests
pytest tests/ -v
```

---

## Dashboard Pages

| Page | What it shows |
|---|---|
| **Overview** | Business KPIs, churn rate, key insights |
| **EDA** | Interactive feature analysis, distributions |
| **Model Results** | ROC/PR curves, confusion matrix, business impact chart |
| **SHAP** | Global importance, beeswarm, single-customer waterfall |
| **Predict** | Real-time churn probability for any customer |

---

## Technical Highlights

- **SMOTE** for class imbalance: oversamples minority class in training only (no data leakage)
- **MLflow tracking**: every experiment logged with params, metrics, and model artifacts
- **SHAP TreeExplainer**: O(n·d) vs brute-force O(2^d) — fast enough for production use
- **Business impact chart**: finds optimal threshold to maximise net revenue recovery
- **CI pipeline**: GitHub Actions runs tests + model quality gate on every push
- **imblearn Pipeline**: SMOTE correctly integrated so it never runs at inference time

---

## Dataset

IBM Telco Customer Churn — 7,043 customers, 21 features, 26.5% churn rate.
Available on [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn).

---

## Author

**Pratham Patil** — [prathampatil.me](https://www.prathampatil.me) ·
[GitHub](https://github.com/PrathamPatil17) ·
[LinkedIn](https://linkedin.com/in/prathamlpatil)
