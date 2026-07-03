# Day-by-Day Build Guide — Customer Churn Prediction

Total time: 10–12 days | Goal: interview-ready, industry-grade portfolio project

---

## Day 1 — Setup and Dataset

**Time: 2–3 hours**

```bash
# 1. Create project
mkdir churn-prediction && cd churn-prediction
git init
pip install -r requirements.txt

# 2. Get dataset
# Go to: https://www.kaggle.com/datasets/blastchar/telco-customer-churn
# Download CSV → place at data/WA_Fn-UseC_-Telco-Customer-Churn.csv

# 3. First look
python -c "
import pandas as pd
df = pd.read_csv('data/WA_Fn-UseC_-Telco-Customer-Churn.csv')
print(df.shape, df.dtypes, df['Churn'].value_counts())
"
```

**What to observe:**
- 7,043 rows × 21 columns
- TotalCharges is object dtype (whitespace bug — you fix this in data_loader.py)
- 26.5% churn rate — moderately imbalanced, needs SMOTE

**Commit:** `git commit -m "feat: initial project setup and dataset"`

---

## Day 2 — EDA (The Most Important Day)

**Time: 3–4 hours**

```bash
python notebooks/01_eda.py
```

**5 findings you will quote in every interview:**

1. Month-to-month customers: **42% churn rate** vs 11% (1-year) vs 3% (2-year)
2. Churned customers' avg tenure: **17 months** vs 38 months for retained
3. Fiber optic customers churn more (30%) than DSL (20%) — despite paying more
4. Customers with no tech support: **41% churn** vs 15% with tech support
5. Senior citizens churn at **42%** vs 24% overall

**The interview answer this gives you:**
> "Before touching a model, I did EDA and found that contract type explained
> more variance in churn than any other variable. That immediately told me the
> business intervention should focus on moving customers from M2M to annual
> contracts — the model just needs to identify who to target."

**Commit:** `git commit -m "feat: EDA with 5 business insights"`

---

## Day 3 — Data Pipeline

**Time: 2 hours**

Run and review src/data_loader.py. Understand each function:

- `clean()` — fixes TotalCharges dtype, encodes target, drops customerID
- `encode()` — binary encodes yes/no cols, one-hot encodes multi-class
- `engineer_features()` — creates 5 business-driven features:
  - `charges_per_tenure` — monthly burn rate signal
  - `service_count` — adoption breadth (more services = more sticky)
  - `long_term_contract` — strongest single churn reducer
  - `monthly_to_total_ratio` — new vs established customer signal
  - `tenure_band` — lifecycle stage bucket

**Why feature engineering matters (say this in interviews):**
> "Raw features like TotalCharges are correlated with tenure, making it hard
> for the model to isolate the cost signal. charges_per_tenure separates these —
> a customer paying $80/month for 2 months is much riskier than one paying $80
> for 36 months. The model can't learn this from raw features alone."

**Commit:** `git commit -m "feat: data pipeline with feature engineering"`

---

## Day 4–5 — Model Training with MLflow

**Time: 4–5 hours**

```bash
# Terminal 1: start MLflow UI
mlflow ui --port 5000

# Terminal 2: run training
python notebooks/02_train_and_evaluate.py
```

Visit http://localhost:5000 — you'll see all 4 experiments logged.

**What to look for in MLflow:**
- XGBoost should hit AUC ~0.91
- Check CV AUC vs test AUC — if gap > 0.03, overfitting
- Compare precision vs recall trade-off by model

**The SMOTE interview question:**
> Q: "Didn't SMOTE cause data leakage?"
> A: "No — I wrapped SMOTE inside an imblearn Pipeline, so it only runs
> during .fit() on training folds. It never sees test data, and during
> predict() it's skipped entirely. This is the correct way to do it."

**Commit:** `git commit -m "feat: train 4 models with MLflow tracking and SMOTE"`

---

## Day 6 — Evaluation Deep Dive

**Time: 2–3 hours**

Key evaluation decisions to understand and explain:

**Why ROC-AUC over accuracy?**
- With 26.5% churn rate, a model predicting "never churn" gets 73.5% accuracy
- ROC-AUC measures discrimination ability regardless of threshold
- PR-AUC is even better for imbalanced — look at this too

**Why threshold = 0.38, not 0.5?**
- Default 0.5 optimises accuracy
- Business cares about net revenue: catching true churners (recall) vs
  wasting intervention budget on non-churners (precision)
- Run business_impact_chart() to find the optimal threshold for your cost assumptions

**The confusion matrix conversation:**
- False Negative (missed churner) = ~$1,200 lost
- False Positive (wrongly targeted non-churner) = $50 wasted
- Asymmetric costs → lower threshold, favour recall

**Commit:** `git commit -m "feat: evaluation suite with business impact analysis"`

---

## Day 7–8 — SHAP Explainability

**Time: 3–4 hours**

```bash
python notebooks/03_shap_analysis.py
```

**The 4 SHAP questions you will be asked:**

Q: "What is SHAP?"
> "SHAP assigns each feature a contribution value for each prediction,
> based on game theory (Shapley values). Unlike feature importance from
> trees, SHAP values are consistent — a feature with high SHAP importance
> always has high impact, regardless of model structure."

Q: "How is SHAP different from feature importance?"
> "Tree feature importance counts split frequency — a feature used early
> in many trees looks important even if its actual prediction impact is
> small. SHAP measures the actual magnitude of impact on each prediction,
> so it's more trustworthy and more actionable."

Q: "How do you use SHAP in production?"
> "For flagged high-risk customers, we generate a waterfall chart showing
> the top 5 drivers. The retention team can then personalise the outreach —
> if the top driver is 'no tech support', offer a free tech support trial
> rather than a generic discount."

Q: "What did SHAP tell you that the model didn't?"
> "Tenure has a strongly non-linear effect — the risk drops sharply after
> month 12 and again after month 24. A linear model would miss this. The
> SHAP dependence plot showed the exact inflection points."

**Commit:** `git commit -m "feat: SHAP global and local explainability"`

---

## Day 9 — Streamlit Dashboard

**Time: 3–4 hours**

```bash
streamlit run dashboard/app.py
```

**5 pages your interviewer will click through:**
1. Overview → business KPIs at a glance
2. EDA → interactive feature exploration
3. Model Results → curves, confusion matrix, business impact
4. SHAP → global importance + single customer waterfall
5. Predict → enter any customer → get probability + explanation

**Polish checklist:**
- [ ] All charts load in < 3 seconds (use @st.cache_data)
- [ ] Business impact chart has adjustable sliders
- [ ] Single prediction page gives a retention recommendation
- [ ] Mobile layout doesn't break

**Commit:** `git commit -m "feat: 5-page Streamlit dashboard with live prediction"`

---

## Day 10 — Tests, CI, Polish

**Time: 2–3 hours**

```bash
# Run tests
pytest tests/ -v

# Check all tests pass
# Push to GitHub — CI pipeline runs automatically
git push origin main
```

**README polish checklist:**
- [ ] Results table with actual numbers filled in
- [ ] Live dashboard link (deploy to Streamlit Community Cloud — free)
- [ ] Business framing in first paragraph
- [ ] Architecture diagram
- [ ] Quick start works from scratch in < 5 mins

**Deploy dashboard (free):**
1. Go to share.streamlit.io
2. Connect GitHub repo
3. Set main file: dashboard/app.py
4. Add dataset via Streamlit Secrets or commit a small sample

**Commit:** `git commit -m "feat: tests, CI pipeline, README polish"`

---

## How to Talk About This Project in Interviews

### 2-minute elevator pitch:
> "I built a customer churn prediction system on the IBM Telco dataset —
> 7,000 customers, 21 features, 26.5% churn rate. After EDA I found that
> contract type and tenure were the dominant signals, which shaped the
> feature engineering. I trained four models — logistic regression through
> XGBoost — with full MLflow tracking and SMOTE for class imbalance.
> XGBoost hit ROC-AUC of 0.91.
>
> The part I'm most proud of is the business layer: instead of using
> default threshold 0.5, I built a business impact calculator that finds
> the optimal threshold given cost per intervention and customer value.
> At $50 per intervention and $1,200 annual value, the optimal threshold
> is 0.38, and the model recovers ~$142,000 in net revenue on the test set.
>
> I also added SHAP explainability so the retention team knows WHY each
> customer is flagged — not just that they're at risk. That's what makes
> it actionable."

### When they ask "what would you improve?":
> "Three things: First, I'd add a data drift monitor — if the distribution
> of MonthlyCharges shifts next quarter, the model needs retraining signals.
> Second, I'd experiment with survival analysis (Cox PH model) to predict
> not just IF but WHEN a customer will churn. Third, I'd A/B test the
> retention interventions themselves — the model tells us who to target,
> but we need to measure which interventions actually work."

---

## Deployment (Optional — adds significant resume value)

```bash
# Streamlit Community Cloud (free)
# 1. Push code to GitHub
# 2. share.streamlit.io → New app → select repo → dashboard/app.py

# Or Docker
docker build -t churn-dashboard .
docker run -p 8501:8501 churn-dashboard
```
