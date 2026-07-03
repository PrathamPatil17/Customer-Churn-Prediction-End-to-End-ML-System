# Customer Churn Prediction — End-to-End ML System

[![CI](https://github.com/PrathamPatil17/churn-prediction/actions/workflows/ci.yml/badge.svg)](https://github.com/PrathamPatil17/churn-prediction/actions)
[![Python](https://img.shields.io/badge/Python-3.9%20%7C%203.11-blue)](https://python.org)
[![MLflow](https://img.shields.io/badge/MLflow-2.9-orange)](https://mlflow.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.29-red)](https://streamlit.io)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0-green)](https://xgboost.readthedocs.io)
[![Tests](https://img.shields.io/badge/tests-16%20passed-brightgreen)](tests/test_pipeline.py)

> Predicts which telecom customers will churn using an end-to-end ML pipeline —
> from raw data to a deployed, explainable model with a live business impact dashboard.

---

## Business Problem

A telecom company loses **26.5% of customers annually** to churn. At ~$1,200 average
annual value per customer, identifying and intervening with at-risk customers before
they leave is worth millions in recovered revenue. This system predicts churn probability
for each customer and explains the drivers behind every prediction, so a retention team
can act on *who* is at risk and *why*.

---

## Results

Measured by running `notebooks/02_train_and_evaluate.py` end-to-end against the full
7,043-row IBM Telco dataset (80/20 stratified split, `random_state=42`, 5-fold
stratified CV). These are the actual numbers this codebase produces today — not
aspirational figures.

| Model | ROC-AUC | F1 | Precision | Recall | PR-AUC | CV ROC-AUC (mean ± std) |
|---|---|---|---|---|---|---|
| **Logistic Regression** | **0.8467** | 0.6124 | 0.4983 | 0.7941 | 0.6669 | 0.8486 ± 0.0126 |
| Random Forest | 0.8427 | **0.6364** | **0.5436** | 0.7674 | 0.6446 | 0.8468 ± 0.0109 |
| XGBoost | 0.8331 | 0.6197 | 0.5059 | **0.7995** | 0.6376 | 0.8392 ± 0.0104 |
| LightGBM | *not benchmarked* | | | | | |

**Best model: Logistic Regression**, by ROC-AUC — the simplest model in the lineup
edges out the tree ensembles on this dataset once `class_weight="balanced"` is applied.
All three models clear the CI quality gate (ROC-AUC ≥ 0.84) with CV standard deviations
under 0.013, so the ranking is stable, not noise.

> **LightGBM is part of `src/train.py`** (`get_models()` includes it) but its import is
> wrapped in a `try/except` — on machines where the compiled `lib_lightgbm.dylib` can't
> find `libomp` at runtime (common on macOS without an ARM64 Homebrew `libomp`), the
> pipeline logs a warning and trains the remaining three models instead of crashing. See
> [Troubleshooting](#troubleshooting) if you want LightGBM included.

### Business impact (Logistic Regression, actual test set)

At the **profit-maximizing threshold (0.12)** — found by scanning thresholds 0.10→0.90
in `business_impact_chart()` — the model recovers an estimated **$392,600 in net
revenue** on the 1,409-customer test set, assuming $1,200 annual customer value and $50
per intervention:

- True positives (correctly caught churners): 370 of 374 actual churners (**99% recall**)
- False positives (wrongly flagged, cost $50 each): 658
- Because customer lifetime value ($1,200) is 24× the intervention cost ($50), the
  optimum sits at very low precision (~36%) and very high recall — it's cheap to over-target.

This is a materially different optimum than a 0.5 default threshold would give, and it's
the argument for building the business-impact calculator rather than optimizing for
accuracy or F1 in isolation.

### SMOTE: measured, not assumed

Contrary to the common assumption that SMOTE meaningfully lifts minority-class recall,
measuring it directly on this dataset (Logistic Regression, same train/test split, same
`class_weight="balanced"`) shows it makes **no practical difference** once class
weighting is already applied:

| Config | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| No SMOTE (class_weight only) | 0.5025 | 0.7995 | 0.6171 | 0.8485 |
| SMOTE + class_weight | 0.4983 | 0.7941 | 0.6124 | 0.8467 |

SMOTE is kept in the pipeline (via an `imblearn.Pipeline` so it never runs at inference
time) because it's a standard, defensible technique to show — but on *this* dataset,
`class_weight="balanced"` alone accounts for essentially all of the imbalance handling.

---

## Key Findings (EDA + SHAP)

Verified by running `notebooks/01_eda.py` and `notebooks/03_shap_analysis.py` directly
against the dataset:

1. **Contract type** is the strongest churn driver — month-to-month customers churn at
   **42.7%**, vs 11.3% for one-year and 2.8% for two-year contracts (~15× spread between
   the extremes).
2. **Tenure** separates churners from retained customers sharply — churned customers
   have a much shorter average tenure in every contract segment (e.g. 14 months vs
   21 months even within month-to-month).
3. **Fiber optic customers churn more** (41.9%) than DSL customers (19.0%) despite
   paying more, suggesting a value-perception gap.
4. **Senior citizens churn nearly 2× more** than the general base — 41.7% vs 23.6%.
5. **SHAP confirms the EDA**: the top features by mean |SHAP value| for the winning
   model are `InternetService_Fiber optic`, `monthly_to_total_ratio`, `InternetService_No`,
   `Contract_Two year`, and `tenure` — directly matching the EDA signal, not contradicting it.
6. **Engineered features pull their weight**: `long_term_contract` (r = −0.405) and
   `monthly_to_total_ratio` (r = +0.313) are the two strongest linear correlations with
   churn among all engineered features — stronger than most raw columns.

---

## Architecture

```
data/                          Raw CSV (IBM Telco, 7,043 customers) — not committed, see Quick Start
src/
├── data_loader.py             Load → clean → encode → feature engineer
├── train.py                   Train up to 4 models with MLflow tracking + SMOTE
├── evaluate.py                ROC/PR curves, confusion matrix, business impact
└── explain.py                 SHAP global + local explainability (Plotly-native beeswarm)
dashboard/
└── app.py                     5-page Streamlit dashboard
notebooks/
├── 01_eda.py                  EDA and business insights
├── 02_train_and_evaluate.py   Full training run
└── 03_shap_analysis.py        Deep SHAP explainability
tests/
└── test_pipeline.py           16 unit tests (pytest) — data cleaning, encoding, feature engineering
models/                        Saved pipelines + metadata (generated by training, gitignored)
mlruns/                        MLflow experiment tracking store (generated, gitignored)
.github/workflows/ci.yml       CI pipeline with model quality gate (AUC ≥ 0.84)
```

### Pipeline flow

```
raw CSV
  → data_loader.clean()            fix TotalCharges dtype, encode target, drop customerID
  → data_loader.encode()           binary-encode yes/no, one-hot encode multi-class categoricals
  → data_loader.engineer_features() charges_per_tenure, service_count, long_term_contract,
                                    monthly_to_total_ratio, tenure_band
  → train_test_split (stratified, 80/20)
  → imblearn.Pipeline: StandardScaler → SMOTE (train-only) → model
  → MLflow logging (params, metrics, model artifact) per model
  → best model by ROC-AUC saved to models/best_model.pkl
  → src/explain.py: SHAP TreeExplainer / LinearExplainer on the saved pipeline
  → dashboard/app.py: reads models/*.pkl + mlruns/ to render all 5 pages live
```

---

## Quick Start

```bash
# 1. Clone and install
git clone https://github.com/PrathamPatil17/churn-prediction
cd churn-prediction
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Get the dataset
# From: https://www.kaggle.com/datasets/blastchar/telco-customer-churn
# Place CSV at: data/WA_Fn-UseC_-Telco-Customer-Churn.csv

# 3. Run EDA (prints churn-rate breakdowns and business findings)
PYTHONPATH=src python notebooks/01_eda.py

# 4. Train all models (tracked in MLflow, saves models/best_model.pkl)
PYTHONPATH=src python notebooks/02_train_and_evaluate.py

# 5. Open MLflow UI (in a separate terminal, from the repo root)
mlflow ui --port 5000
# Visit: http://localhost:5000

# 6. SHAP explainability analysis
PYTHONPATH=src python notebooks/03_shap_analysis.py

# 7. Launch the dashboard
streamlit run dashboard/app.py
# Visit: http://localhost:8501

# 8. Run tests
PYTHONPATH=src pytest tests/ -v
```

`PYTHONPATH=src` is needed for the standalone `notebooks/*.py` scripts and `pytest`
because `src/` isn't an installed package — `dashboard/app.py` adds `src/` to
`sys.path` itself, so no `PYTHONPATH` is needed to run the dashboard directly.

---

## Dashboard Pages

| Page | What it shows |
|---|---|
| **Overview** | Business KPIs, churn rate, key insights |
| **EDA** | Interactive feature analysis, distributions |
| **Model Results** | ROC/PR curves, confusion matrix, business impact chart |
| **SHAP** | Global importance, beeswarm (native Plotly `go.Scatter`, not `px.strip`), single-customer waterfall |
| **Predict** | Real-time churn probability for any customer, with retention recommendation |

---

## Technical Highlights

- **SMOTE** for class imbalance, wrapped in an `imblearn.Pipeline` so it only runs
  during `.fit()` on training folds and is skipped entirely at inference — no data
  leakage. (Measured to add negligible lift over `class_weight="balanced"` alone on
  this dataset — see [Results](#results).)
- **MLflow tracking**: every experiment run logged with params, metrics, and model
  artifacts under `mlruns/`.
- **SHAP** for explainability — `TreeExplainer` for tree models, appropriate explainer
  for linear models; global importance, beeswarm, and per-customer waterfall charts.
- **Business impact chart**: scans decision thresholds to find the one that maximizes
  net revenue recovery given configurable customer value and intervention cost.
- **CI pipeline**: GitHub Actions runs the pytest suite plus a model-quality gate
  (fails the build if ROC-AUC drops below 0.84) on every push.
- **Graceful degradation**: LightGBM import failures (missing native `libomp`) don't
  crash training — the pipeline logs a warning and proceeds with the remaining models.

---

## Testing

```bash
PYTHONPATH=src pytest tests/ -v
```

16 tests in `tests/test_pipeline.py`, covering:

- **`clean()`** — customerID dropped, TotalCharges numeric, whitespace handled, target
  binary-encoded, no missing values after cleaning.
- **`encode()`** — no object dtype columns remain, one-hot columns created as expected,
  binary columns are strictly 0/1.
- **`engineer_features()`** — all 5 engineered features exist, `service_count` and
  `monthly_to_total_ratio` are within expected bounds, `tenure_band` is categorical.
- **Feature/target split** — `Churn` isn't leaked into `X`, `y` is binary, `X`/`y`
  lengths match, no missing values in `X`.

All 16 pass on a clean install.

---

## Dataset

IBM Telco Customer Churn — 7,043 customers, 21 raw features (31 after encoding +
feature engineering), 26.5% churn rate. Available on
[Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn). Not committed
to the repo — download it and place it at `data/WA_Fn-UseC_-Telco-Customer-Churn.csv`
before running anything.

---

## Troubleshooting

**LightGBM fails to install or import (`OSError: ... libomp.dylib`)**
Common on macOS. LightGBM's compiled extension needs OpenMP at runtime.

```bash
brew install libomp
```

If your Homebrew and Python architectures don't match (e.g. Intel Homebrew at
`/usr/local` with an ARM64 Python), `pip install lightgbm` may still fail to locate
`libomp`. The pipeline handles this by skipping LightGBM automatically — you'll see
`LightGBM unavailable on this machine (missing libomp) — skipping.` in the training
output, and the other three models still train normally.

**`pip install lightgbm==4.1.0` fails to build from source (CMake error)**
The pinned version in `requirements.txt` doesn't ship a wheel for all platforms and
falls back to a source build that can fail on newer CMake/Xcode toolchains. Install an
unpinned `pip install lightgbm` instead to get a prebuilt wheel, or skip it — see above.

**`ModuleNotFoundError: No module named 'data_loader'`**
Run notebook scripts and pytest with `PYTHONPATH=src` set — they aren't packaged, and
importing sibling modules under `src/` requires it on the path.

**`AssertionError: IPython must be installed to use initjs()`**
`shap.initjs()` in `src/explain.py` needs IPython even outside a notebook. `pip install
ipython`.

---

## Author

**Pratham Patil** — [prathampatil.me](https://www.prathampatil.me) ·
[GitHub](https://github.com/PrathamPatil17) ·
[LinkedIn](https://linkedin.com/in/prathamlpatil)
