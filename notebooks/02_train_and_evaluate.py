"""
notebooks/02_train_and_evaluate.py
------------------------------------
Day 4–6: Train all models, evaluate, compare.
Run MLflow UI in parallel: mlflow ui
"""

import sys
sys.path.insert(0, "../src")

from data_loader import load_pipeline
from train import train_all, get_metrics_table
from evaluate import (roc_curve_comparison, pr_curve_comparison,
                       confusion_matrix_chart, business_impact_chart,
                       model_comparison_table)

# ── 1. Load processed data ────────────────────────────────────────────────────
print("Loading and preprocessing data...")
X, y, feature_names = load_pipeline()
print(f"Features: {X.shape[1]} | Samples: {len(X):,} | Churn rate: {y.mean():.1%}")

# ── 2. Train all models with MLflow tracking ──────────────────────────────────
print("\nStarting training run...")
print("Open MLflow UI: mlflow ui --port 5000")
results, X_test, y_test = train_all(X, y, experiment_name="churn_prediction_v1")

# ── 3. Print comparison table ─────────────────────────────────────────────────
print("\n=== MODEL COMPARISON ===")
print(get_metrics_table(results).to_string())

# ── 4. Best model analysis ────────────────────────────────────────────────────
best_name = max(results, key=lambda k: results[k]["metrics"]["roc_auc"])
best = results[best_name]

print(f"\n=== BEST MODEL: {best_name} ===")
print(f"ROC-AUC:   {best['metrics']['roc_auc']:.4f}")
print(f"F1 Score:  {best['metrics']['f1']:.4f}")
print(f"Precision: {best['metrics']['precision']:.4f}")
print(f"Recall:    {best['metrics']['recall']:.4f}")
print(f"CV AUC:    {best['metrics']['cv_roc_auc_mean']:.4f} ± {best['metrics']['cv_roc_auc_std']:.4f}")

# ── 5. Business impact ────────────────────────────────────────────────────────
_, best_thresh = business_impact_chart(
    best["y_test"], best["y_prob"],
    avg_customer_value=1200, cost_per_intervention=50
)
print(f"\nOptimal decision threshold: {best_thresh:.2f}")
print("(Use this threshold, not default 0.5, for maximum business impact)")

print("\n✅ Training complete. Next steps:")
print("  1. Run: mlflow ui  to browse experiments")
print("  2. Run: python src/explain.py  for SHAP analysis")
print("  3. Run: streamlit run dashboard/app.py  for the dashboard")
