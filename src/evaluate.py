"""
src/evaluate.py
---------------
Generates all model evaluation charts:
  - ROC curve comparison
  - Precision-Recall curve comparison
  - Confusion matrix
  - Calibration curve
  - Business impact chart (revenue saved by catching churners)
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.metrics import (
    roc_curve, auc,
    precision_recall_curve,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.calibration import calibration_curve


COLORS = {
    "LogisticRegression": "#888780",
    "RandomForest": "#2a78d6",
    "XGBoost": "#E24B4A",
    "LightGBM": "#1baf7a",
}


def roc_curve_comparison(results: dict) -> go.Figure:
    """Overlay ROC curves for all models."""
    fig = go.Figure()

    for name, r in results.items():
        fpr, tpr, _ = roc_curve(r["y_test"], r["y_prob"])
        auc_score = auc(fpr, tpr)
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr,
            name=f"{name} (AUC={auc_score:.3f})",
            mode="lines",
            line=dict(color=COLORS.get(name, "#333"), width=2),
        ))

    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        name="Random classifier",
        mode="lines",
        line=dict(dash="dash", color="gray", width=1),
        showlegend=True,
    ))

    fig.update_layout(
        title="ROC Curve — Model Comparison",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        height=420,
        legend=dict(x=0.55, y=0.15),
        margin=dict(t=50, b=10, l=10, r=10),
    )
    return fig


def pr_curve_comparison(results: dict) -> go.Figure:
    """Overlay Precision-Recall curves. More informative than ROC for imbalanced data."""
    fig = go.Figure()

    for name, r in results.items():
        prec, rec, _ = precision_recall_curve(r["y_test"], r["y_prob"])
        pr_auc = auc(rec, prec)
        fig.add_trace(go.Scatter(
            x=rec, y=prec,
            name=f"{name} (PR-AUC={pr_auc:.3f})",
            mode="lines",
            line=dict(color=COLORS.get(name, "#333"), width=2),
        ))

    baseline = results[list(results.keys())[0]]["y_test"].mean()
    fig.add_hline(y=baseline, line_dash="dash", line_color="gray",
                   annotation_text=f"Baseline ({baseline:.2f})")

    fig.update_layout(
        title="Precision-Recall Curve — Model Comparison",
        xaxis_title="Recall",
        yaxis_title="Precision",
        height=420,
        legend=dict(x=0.35, y=0.95),
        margin=dict(t=50, b=10, l=10, r=10),
    )
    return fig


def confusion_matrix_chart(y_true, y_pred, model_name: str = "") -> go.Figure:
    """Annotated confusion matrix as a heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    labels = ["Retained", "Churned"]

    fig = px.imshow(
        cm,
        labels=dict(x="Predicted", y="Actual", color="Count"),
        x=labels, y=labels,
        color_continuous_scale=["#E6F1FB", "#185FA5"],
        text_auto=True,
        aspect="auto",
        title=f"Confusion Matrix — {model_name}",
    )
    fig.update_layout(
        height=340,
        margin=dict(t=50, b=10, l=10, r=10),
        coloraxis_showscale=False,
    )
    return fig


def metrics_radar(results: dict) -> go.Figure:
    """Radar chart comparing models across all metrics."""
    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    fig = go.Figure()

    for name, r in results.items():
        values = [r["metrics"][m] for m in metrics]
        values.append(values[0])  # close the polygon
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=metrics + [metrics[0]],
            fill="toself",
            name=name,
            line_color=COLORS.get(name, "#333"),
            opacity=0.6,
        ))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0.5, 1.0])),
        title="Model Comparison — Metric Radar",
        height=420,
        showlegend=True,
        margin=dict(t=50, b=10, l=30, r=30),
    )
    return fig


def business_impact_chart(y_true, y_prob,
                            avg_customer_value: float = 1200.0,
                            cost_per_intervention: float = 50.0) -> go.Figure:
    """
    This is the chart that impresses business-minded interviewers.

    Shows: at each probability threshold, how much revenue is saved
    by targeting predicted churners with a retention campaign.

    Assumptions:
      - avg_customer_value: annual revenue per retained customer ($)
      - cost_per_intervention: cost of reaching out to one customer ($)
    """
    thresholds = np.arange(0.1, 0.9, 0.02)
    net_gains = []
    precisions = []
    recalls = []

    n_positive = y_true.sum()

    for thresh in thresholds:
        y_pred = (y_prob >= thresh).astype(int)
        tp = ((y_pred == 1) & (y_true == 1)).sum()
        fp = ((y_pred == 1) & (y_true == 0)).sum()

        revenue_saved = tp * avg_customer_value
        intervention_cost = (tp + fp) * cost_per_intervention
        net_gain = revenue_saved - intervention_cost

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / n_positive if n_positive > 0 else 0

        net_gains.append(net_gain)
        precisions.append(prec)
        recalls.append(rec)

    best_idx = np.argmax(net_gains)
    best_thresh = thresholds[best_idx]
    best_gain = net_gains[best_idx]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=thresholds, y=net_gains,
        mode="lines",
        name="Net revenue impact ($)",
        line=dict(color="#2a78d6", width=2),
        fill="tozeroy",
        fillcolor="rgba(42,120,214,0.1)",
    ))
    fig.add_vline(
        x=best_thresh,
        line_dash="dash",
        line_color="#E24B4A",
        annotation_text=f"Optimal threshold: {best_thresh:.2f}<br>Net gain: ${best_gain:,.0f}",
        annotation_position="top right",
    )
    fig.update_layout(
        title=f"Business Impact: Revenue Saved by Targeting Predicted Churners<br>"
              f"<sup>Assumptions: ${avg_customer_value}/customer/yr, "
              f"${cost_per_intervention}/intervention</sup>",
        xaxis_title="Probability Threshold",
        yaxis_title="Net Revenue Impact ($)",
        height=400,
        margin=dict(t=70, b=10, l=10, r=10),
    )
    return fig, best_thresh


def model_comparison_table(results: dict) -> pd.DataFrame:
    """Clean DataFrame of all models × all metrics, sorted by ROC-AUC."""
    rows = []
    for name, r in results.items():
        row = {"Model": name}
        for k, v in r["metrics"].items():
            row[k.upper().replace("_", " ")] = v
        rows.append(row)
    return pd.DataFrame(rows).set_index("Model").sort_values("ROC AUC", ascending=False)
