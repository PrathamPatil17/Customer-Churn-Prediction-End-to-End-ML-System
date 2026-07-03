"""
src/explain.py
--------------
SHAP-based model explainability.
Works with any tree-based model in a sklearn Pipeline.
Produces figures consumable by both notebooks and Streamlit.
"""

import warnings
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import shap
import joblib
from pathlib import Path

warnings.filterwarnings("ignore")
shap.initjs()


def get_explainer(pipeline, X_sample: pd.DataFrame):
    """
    Build the right SHAP explainer for the model inside the pipeline.
    Extracts the actual model and applies any scaling first.
    """
    model = pipeline.named_steps["model"]
    model_name = type(model).__name__

    # Get the transformed X (after scaler, before model)
    steps_before_model = list(pipeline.named_steps.keys())[:-1]
    # Filter out SMOTE — it only runs during fit, not transform
    transform_steps = [
        (k, v) for k, v in pipeline.named_steps.items()
        if k != "model" and hasattr(v, "transform")
    ]

    X_transformed = X_sample.copy()
    for step_name, step in transform_steps:
        X_transformed = pd.DataFrame(
            step.transform(X_transformed),
            columns=X_sample.columns,
            index=X_sample.index,
        )

    if model_name in ["XGBClassifier", "LGBMClassifier", "RandomForestClassifier",
                       "GradientBoostingClassifier"]:
        explainer = shap.TreeExplainer(model)
    else:
        explainer = shap.LinearExplainer(model, X_transformed)

    return explainer, X_transformed


def compute_shap_values(pipeline, X_sample: pd.DataFrame):
    """Compute SHAP values. Returns (explainer, shap_values, X_transformed)."""
    explainer, X_transformed = get_explainer(pipeline, X_sample)
    shap_values = explainer.shap_values(X_transformed)

    # For classifiers that return a list (RF), take the positive class
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    return explainer, shap_values, X_transformed


def global_importance_chart(shap_values: np.ndarray,
                             X: pd.DataFrame,
                             top_n: int = 15) -> go.Figure:
    """
    Bar chart of mean absolute SHAP values (global feature importance).
    This is what you show first in every DS interview.
    """
    mean_abs = np.abs(shap_values).mean(axis=0)
    feat_imp = pd.DataFrame({
        "feature": X.columns,
        "importance": mean_abs,
    }).sort_values("importance", ascending=True).tail(top_n)

    fig = px.bar(
        feat_imp,
        x="importance",
        y="feature",
        orientation="h",
        title=f"Top {top_n} Features by Mean |SHAP Value|",
        labels={"importance": "Mean |SHAP Value|", "feature": "Feature"},
        color="importance",
        color_continuous_scale=["#B5D4F4", "#185FA5"],
    )
    fig.update_layout(
        height=420,
        coloraxis_showscale=False,
        margin=dict(t=50, b=10, l=10, r=10),
        yaxis=dict(tickfont=dict(size=11)),
    )
    return fig


def beeswarm_data(shap_values: np.ndarray,
                  X: pd.DataFrame,
                  top_n: int = 12) -> go.Figure:
    """
    Plotly approximation of SHAP beeswarm plot.
    Shows both direction and magnitude of feature impact.
    """
    top_features = (
        pd.Series(np.abs(shap_values).mean(axis=0), index=X.columns)
        .nlargest(top_n)
        .index.tolist()
    )

    rng = np.random.default_rng(42)

    fig = go.Figure()
    for i, feat in enumerate(top_features):
        idx = list(X.columns).index(feat)
        sv = shap_values[:, idx]
        fv = X[feat].values
        # Normalise feature value 0-1 for colour encoding
        fv_norm = (fv - fv.min()) / (fv.max() - fv.min() + 1e-9)
        jitter = rng.uniform(-0.35, 0.35, size=len(sv))

        fig.add_trace(go.Scatter(
            x=sv,
            y=[i + j for j in jitter],
            mode="markers",
            marker=dict(
                size=4,
                opacity=0.6,
                color=fv_norm,
                colorscale=["#185FA5", "#E24B4A"],
                showscale=(i == 0),
                colorbar=dict(
                    title="Feature value",
                    tickvals=[0, 1],
                    ticktext=["Low", "High"],
                ) if i == 0 else None,
            ),
            name=feat,
            hovertemplate=f"{feat}<br>SHAP: %{{x:.3f}}<extra></extra>",
            showlegend=False,
        ))

    fig.update_layout(
        title="Feature Impact on Churn Prediction (SHAP Beeswarm)",
        xaxis_title="SHAP value (impact on model output)",
        yaxis=dict(
            tickmode="array",
            tickvals=list(range(len(top_features))),
            ticktext=top_features,
            tickfont=dict(size=11),
        ),
        height=460,
        margin=dict(t=50, b=10, l=10, r=10),
    )
    return fig


def single_prediction_waterfall(explainer, shap_values: np.ndarray,
                                  X: pd.DataFrame,
                                  sample_idx: int = 0) -> go.Figure:
    """
    Waterfall chart for a single customer prediction.
    Shows exactly why the model predicted churn / no-churn for that person.
    Great for the 'how do you make it actionable?' interview question.
    """
    sv = shap_values[sample_idx]
    base = explainer.expected_value
    if isinstance(base, np.ndarray):
        base = base[1]

    # Take top contributing features
    feat_df = pd.DataFrame({
        "feature": X.columns,
        "shap": sv,
        "value": X.iloc[sample_idx].values,
    }).reindex(pd.Series(np.abs(sv), index=X.columns).nlargest(12).index)

    feat_df = feat_df.sort_values("shap")
    colors = ["#E24B4A" if v > 0 else "#2a78d6" for v in feat_df["shap"]]

    fig = go.Figure(go.Bar(
        x=feat_df["shap"],
        y=[f"{r['feature']} = {r['value']:.2f}" for _, r in feat_df.iterrows()],
        orientation="h",
        marker_color=colors,
    ))
    fig.add_vline(x=0, line_width=1, line_color="gray")
    fig.update_layout(
        title=f"Why did the model predict this? (Customer #{sample_idx})<br>"
              f"<sup>Base value: {base:.3f} | Red = pushes toward churn | Blue = pushes away</sup>",
        xaxis_title="SHAP value",
        height=420,
        margin=dict(t=70, b=10, l=10, r=10),
    )
    return fig


def dependence_plot(shap_values: np.ndarray,
                     X: pd.DataFrame,
                     feature: str) -> go.Figure:
    """
    SHAP dependence plot for a specific feature.
    Shows non-linear relationships that linear models miss.
    """
    idx = list(X.columns).index(feature)
    sv = shap_values[:, idx]
    fv = X[feature].values

    fig = px.scatter(
        x=fv, y=sv,
        labels={"x": feature, "y": f"SHAP value for {feature}"},
        title=f"SHAP Dependence Plot: {feature}",
        color=sv,
        color_continuous_scale=["#2a78d6", "#E24B4A"],
        opacity=0.6,
    )
    fig.add_hline(y=0, line_dash="dash", line_color="gray", line_width=1)
    fig.update_layout(
        height=360,
        coloraxis_showscale=False,
        margin=dict(t=50, b=10, l=10, r=10),
    )
    return fig


def load_best_model():
    """Load saved best model pipeline."""
    path = Path("models/best_model.pkl")
    if not path.exists():
        raise FileNotFoundError("Run src/train.py first to generate models/best_model.pkl")
    return joblib.load(path)
