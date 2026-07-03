"""
src/eda.py
----------
All EDA functions. Each returns a Plotly figure so they can be
used both in notebooks and in the Streamlit dashboard.
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


PALETTE = {"churn": "#E24B4A", "retain": "#2a78d6", "neutral": "#888780"}


def churn_rate_summary(df: pd.DataFrame) -> go.Figure:
    """Donut chart of overall churn rate with annotation."""
    counts = df["Churn"].value_counts()
    labels = ["Retained", "Churned"]
    values = [counts.get(0, 0), counts.get(1, 0)]
    rate = values[1] / sum(values) * 100

    fig = go.Figure(go.Pie(
        labels=labels,
        values=values,
        hole=0.6,
        marker_colors=[PALETTE["retain"], PALETTE["churn"]],
        textinfo="label+percent",
        showlegend=False,
    ))
    fig.add_annotation(
        text=f"<b>{rate:.1f}%</b><br>Churn Rate",
        x=0.5, y=0.5, showarrow=False,
        font=dict(size=18),
    )
    fig.update_layout(
        title="Overall Churn Rate",
        height=320,
        margin=dict(t=40, b=10, l=10, r=10),
    )
    return fig


def churn_by_feature(df: pd.DataFrame, col: str) -> go.Figure:
    """
    Grouped bar chart: churn rate per category of a feature.
    Works for both categorical and binned numeric columns.
    """
    grouped = (
        df.groupby(col)["Churn"]
        .agg(["sum", "count"])
        .rename(columns={"sum": "churned", "count": "total"})
    )
    grouped["churn_rate"] = grouped["churned"] / grouped["total"] * 100

    fig = px.bar(
        grouped.reset_index(),
        x=col,
        y="churn_rate",
        color="churn_rate",
        color_continuous_scale=["#2a78d6", "#E24B4A"],
        labels={"churn_rate": "Churn Rate (%)"},
        title=f"Churn Rate by {col}",
        text_auto=".1f",
    )
    fig.update_layout(
        height=340,
        coloraxis_showscale=False,
        margin=dict(t=40, b=10),
    )
    return fig


def numeric_distributions(df: pd.DataFrame) -> go.Figure:
    """Overlapping histograms for numeric features split by churn."""
    numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    fig = make_subplots(rows=1, cols=3, subplot_titles=numeric_cols)

    for i, col in enumerate(numeric_cols, 1):
        for label, color, name in [
            (0, PALETTE["retain"], "Retained"),
            (1, PALETTE["churn"], "Churned"),
        ]:
            subset = df[df["Churn"] == label][col].dropna()
            fig.add_trace(
                go.Histogram(
                    x=subset, name=name,
                    marker_color=color, opacity=0.65,
                    showlegend=(i == 1),
                    nbinsx=30,
                ),
                row=1, col=i,
            )

    fig.update_layout(
        barmode="overlay",
        title="Numeric Feature Distributions by Churn",
        height=340,
        legend=dict(orientation="h", y=-0.15),
        margin=dict(t=50, b=10),
    )
    return fig


def correlation_heatmap(df: pd.DataFrame) -> go.Figure:
    """Heatmap of Pearson correlations for numeric features."""
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr().round(2)

    fig = go.Figure(go.Heatmap(
        z=corr.values,
        x=corr.columns,
        y=corr.columns,
        colorscale="RdBu",
        zmid=0,
        text=corr.values.round(2),
        texttemplate="%{text}",
        textfont=dict(size=9),
    ))
    fig.update_layout(
        title="Feature Correlation Matrix",
        height=500,
        margin=dict(t=50, b=10, l=10, r=10),
    )
    return fig


def contract_tenure_churn(df: pd.DataFrame) -> go.Figure:
    """
    Box plot: tenure distribution by contract type, coloured by churn.
    This is the single most important business insight in this dataset.
    """
    contract_col = None
    for col in df.columns:
        if "Contract" in col:
            contract_col = col
            break

    if contract_col is None:
        return go.Figure()

    fig = px.box(
        df, x=contract_col, y="tenure",
        color="Churn",
        color_discrete_map={0: PALETTE["retain"], 1: PALETTE["churn"]},
        labels={"Churn": "Churned", "tenure": "Tenure (months)"},
        title="Tenure Distribution by Contract Type and Churn",
    )
    fig.update_layout(height=360, margin=dict(t=50, b=10))
    return fig


def monthly_charges_boxplot(df: pd.DataFrame) -> go.Figure:
    """Box plot of MonthlyCharges for churned vs retained customers."""
    fig = px.box(
        df, x="Churn", y="MonthlyCharges",
        color="Churn",
        color_discrete_map={0: PALETTE["retain"], 1: PALETTE["churn"]},
        labels={"Churn": "Churned (1=Yes)"},
        title="Monthly Charges: Churned vs Retained",
        points="outliers",
    )
    fig.update_layout(height=340, margin=dict(t=50, b=10))
    return fig


def key_insights(df: pd.DataFrame) -> dict:
    """
    Return a dict of the 5 most important business insights as strings.
    Used in the Streamlit dashboard as metric cards.
    """
    churn_rate = df["Churn"].mean() * 100

    # Month-to-month contract churn rate
    mtm_cols = [c for c in df.columns if "Month-to-month" in c or "Month_to_month" in c]
    if mtm_cols:
        mtm_churn = df[df[mtm_cols[0]] == 1]["Churn"].mean() * 100
    else:
        mtm_churn = None

    # Senior citizen churn
    if "SeniorCitizen" in df.columns:
        senior_churn = df[df["SeniorCitizen"] == 1]["Churn"].mean() * 100
    else:
        senior_churn = None

    # Average tenure of churned customers
    avg_tenure_churned = df[df["Churn"] == 1]["tenure"].mean()

    # Average monthly charges of churned customers
    avg_charges_churned = df[df["Churn"] == 1]["MonthlyCharges"].mean()

    return {
        "Overall Churn Rate": f"{churn_rate:.1f}%",
        "Month-to-Month Contract Churn": f"{mtm_churn:.1f}%" if mtm_churn else "N/A",
        "Senior Citizen Churn Rate": f"{senior_churn:.1f}%" if senior_churn else "N/A",
        "Avg Tenure at Churn (months)": f"{avg_tenure_churned:.1f}",
        "Avg Monthly Charges (Churned)": f"${avg_charges_churned:.2f}",
    }
