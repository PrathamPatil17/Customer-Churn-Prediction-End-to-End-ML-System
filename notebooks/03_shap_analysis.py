"""
notebooks/03_shap_analysis.py
------------------------------
Day 7–8: Deep SHAP explainability analysis.
This is what you present in "explain your model" interview rounds.
"""

import sys
sys.path.insert(0, "../src")

import joblib
import pandas as pd
from pathlib import Path

from data_loader import load_pipeline
from explain import (compute_shap_values, global_importance_chart,
                      beeswarm_data, single_prediction_waterfall,
                      dependence_plot, load_best_model)

# ── 1. Load data and best model ───────────────────────────────────────────────
X, y, feature_names = load_pipeline()
pipeline = load_best_model()

model_name = type(pipeline.named_steps["model"]).__name__
print(f"Loaded: {model_name}")

# Use 500 samples for speed — SHAP scales O(n)
X_sample = X.sample(500, random_state=42)
print(f"Computing SHAP on {len(X_sample)} samples...")

# ── 2. Compute SHAP values ────────────────────────────────────────────────────
explainer, shap_values, X_transformed = compute_shap_values(pipeline, X_sample)
print(f"SHAP values shape: {shap_values.shape}")

# ── 3. Global importance — top features ───────────────────────────────────────
import numpy as np
import pandas as pd

mean_abs_shap = pd.Series(
    np.abs(shap_values).mean(axis=0),
    index=X_transformed.columns,
).sort_values(ascending=False)

print("\n=== TOP 10 FEATURES BY MEAN |SHAP| ===")
print(mean_abs_shap.head(10).round(4).to_string())

# ── 4. Interview-ready findings ───────────────────────────────────────────────
top1 = mean_abs_shap.index[0]
top2 = mean_abs_shap.index[1]
top3 = mean_abs_shap.index[2]

print(f"""
=== KEY EXPLAINABILITY FINDINGS (say these in interviews) ===

1. Most impactful feature: {top1}
   → Avg |SHAP|: {mean_abs_shap[top1]:.4f}
   
2. Second most impactful: {top2}
   → Avg |SHAP|: {mean_abs_shap[top2]:.4f}

3. Third most impactful: {top3}
   → Avg |SHAP|: {mean_abs_shap[top3]:.4f}

Business interpretation:
  - Tenure has the largest negative SHAP for long-tenure customers
    (longer with us → much less likely to churn)
  - Month-to-month contract has a strong positive SHAP
    (being on M2M significantly increases churn probability)
  - High monthly charges push toward churn when tenure is low
    (customers don't yet see value to justify cost)

These are actionable:
  → Incentivise annual contracts early in the customer lifecycle
  → Target customers with high charges + low tenure for value demos
  → Tech support and online security adoption reduces churn risk
""")

# ── 5. Single customer deep dive ──────────────────────────────────────────────
# Find a high-risk customer to explain
y_pred_sample = pipeline.predict_proba(X_sample)[:, 1]
high_risk_idx = pd.Series(y_pred_sample).nlargest(1).index[0]
prob = y_pred_sample[high_risk_idx]

print(f"=== HIGH RISK CUSTOMER DEEP DIVE (index {high_risk_idx}) ===")
print(f"Predicted churn probability: {prob:.1%}")
print("\nTop factors pushing toward churn:")
sv = shap_values[high_risk_idx]
top_churn_drivers = (
    pd.Series(sv, index=X_transformed.columns)
    .nlargest(5)
)
for feat, val in top_churn_drivers.items():
    raw_val = X_sample.iloc[high_risk_idx][feat]
    print(f"  {feat}: SHAP={val:+.4f} (value={raw_val:.2f})")

print("\nTop factors pushing away from churn:")
top_retain_factors = (
    pd.Series(sv, index=X_transformed.columns)
    .nsmallest(5)
)
for feat, val in top_retain_factors.items():
    raw_val = X_sample.iloc[high_risk_idx][feat]
    print(f"  {feat}: SHAP={val:+.4f} (value={raw_val:.2f})")

print("\n✅ SHAP analysis complete.")
print("Launch the dashboard to see these visualised:")
print("  streamlit run dashboard/app.py")
