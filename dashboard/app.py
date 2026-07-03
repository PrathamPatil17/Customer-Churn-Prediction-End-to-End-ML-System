"""
dashboard/app.py
----------------
Streamlit multi-page dashboard for the Churn Prediction project.

Pages:
  1. Overview     — business KPIs and churn summary
  2. EDA          — interactive exploratory analysis
  3. Model Results — performance comparison, ROC/PR curves
  4. SHAP         — explainability charts
  5. Predict      — real-time single customer prediction

Run with: streamlit run dashboard/app.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# Import project modules
from data_loader import load_pipeline, load_raw, clean, encode, engineer_features
from eda import (churn_rate_summary, churn_by_feature, numeric_distributions,
                  correlation_heatmap, monthly_charges_boxplot, key_insights)
from evaluate import (roc_curve_comparison, pr_curve_comparison,
                       confusion_matrix_chart, metrics_radar,
                       business_impact_chart, model_comparison_table)
from explain import compute_shap_values, global_importance_chart, beeswarm_data, single_prediction_waterfall

# ─── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Churn Prediction Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .metric-card {
      background: #f8f9fa;
      border: 1px solid #e9ecef;
      border-radius: 8px;
      padding: 16px 20px;
      text-align: center;
  }
  .metric-value {
      font-size: 28px;
      font-weight: 600;
      color: #1a1a1a;
  }
  .metric-label {
      font-size: 12px;
      color: #6c757d;
      margin-top: 4px;
  }
  .section-header {
      font-size: 18px;
      font-weight: 600;
      color: #1a1a1a;
      margin: 24px 0 12px;
      border-bottom: 2px solid #e9ecef;
      padding-bottom: 8px;
  }
  .insight-box {
      background: #EAF3DE;
      border-left: 4px solid #3B6D11;
      padding: 12px 16px;
      border-radius: 0 8px 8px 0;
      margin: 8px 0;
      font-size: 14px;
      color: #27500A;
  }
  .warning-box {
      background: #FAEEDA;
      border-left: 4px solid #854F0B;
      padding: 12px 16px;
      border-radius: 0 8px 8px 0;
      margin: 8px 0;
      font-size: 14px;
      color: #633806;
  }
</style>
""", unsafe_allow_html=True)


# ─── Data & model loading (cached) ─────────────────────────────────────────────
@st.cache_data(show_spinner="Loading dataset...")
def load_data():
    X, y, feature_names = load_pipeline()
    raw_df = clean(load_raw())
    return X, y, feature_names, raw_df


@st.cache_resource(show_spinner="Loading models...")
def load_models():
    models_dir = Path("models")
    models = {}
    for path in models_dir.glob("*_pipeline.pkl"):
        name = path.stem.replace("_pipeline", "").replace("_", "")
        # Map to display names
        name_map = {
            "logisticregression": "LogisticRegression",
            "randomforest": "RandomForest",
            "xgboost": "XGBoost",
            "lightgbm": "LightGBM",
        }
        display_name = name_map.get(name.lower(), name)
        models[display_name] = joblib.load(path)
    return models


@st.cache_resource(show_spinner="Computing SHAP values (this takes ~30s)...")
def load_shap(_pipeline, X_sample):
    return compute_shap_values(_pipeline, X_sample)


# ─── Sidebar navigation ────────────────────────────────────────────────────────
st.sidebar.title("📊 Churn Prediction")
st.sidebar.markdown("**IBM Telco Customer Churn**")
st.sidebar.divider()

page = st.sidebar.radio(
    "Navigate",
    ["🏠 Overview", "🔍 EDA", "🤖 Model Results", "🧠 SHAP Explainability", "🎯 Predict"],
    label_visibility="collapsed",
)

st.sidebar.divider()
st.sidebar.markdown("""
**Stack**
- XGBoost + LightGBM
- SHAP explainability
- SMOTE class balancing
- MLflow experiment tracking
- Scikit-learn pipelines
""")
st.sidebar.markdown("**[GitHub](https://github.com/PrathamPatil17)** · Built by Pratham Patil")


# ─── Load data ─────────────────────────────────────────────────────────────────
try:
    X, y, feature_names, raw_df = load_data()
    data_loaded = True
except FileNotFoundError as e:
    st.error(str(e))
    data_loaded = False
    st.stop()

models_dir = Path("models")
models_exist = (models_dir / "best_model.pkl").exists()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1: OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Overview":
    st.title("Customer Churn Prediction")
    st.markdown(
        "An end-to-end ML system that identifies customers likely to churn, "
        "explains why, and quantifies the business impact of intervention."
    )

    st.markdown('<div class="section-header">Business KPIs</div>', unsafe_allow_html=True)

    insights = key_insights(raw_df)
    cols = st.columns(len(insights))
    for col, (label, value) in zip(cols, insights.items()):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{value}</div>
                <div class="metric-label">{label}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("")
    col1, col2 = st.columns([1, 2])
    with col1:
        st.plotly_chart(churn_rate_summary(raw_df), use_container_width=True)
    with col2:
        st.plotly_chart(numeric_distributions(raw_df), use_container_width=True)

    st.markdown('<div class="section-header">Key Business Insights</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="insight-box">
        Month-to-month contract customers churn at ~3× the rate of two-year contract customers.
        The single highest-ROI retention action is moving customers from M2M to annual contracts.
    </div>
    <div class="insight-box">
        Customers with tenure under 12 months and high monthly charges represent the highest
        churn risk — they haven't yet seen enough value to justify the cost.
    </div>
    <div class="warning-box">
        26.5% overall churn rate means 1 in 4 customers leaves annually.
        At $1,200 average annual value, every 1% reduction in churn = significant revenue recovery.
    </div>
    """, unsafe_allow_html=True)

    if models_exist:
        meta = joblib.load(models_dir / "metadata.pkl")
        st.markdown('<div class="section-header">Model Performance Summary</div>', unsafe_allow_html=True)
        m1, m2, m3 = st.columns(3)
        m1.metric("Best Model", meta["best_model"])
        m2.metric("ROC-AUC", f"{meta['best_roc_auc']:.4f}")
        m3.metric("Training Samples", f"{meta['train_size']:,}")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2: EDA
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔍 EDA":
    st.title("Exploratory Data Analysis")
    st.markdown(f"Dataset: **{len(raw_df):,} customers** | **{raw_df.shape[1]} features** | "
                f"Churn rate: **{raw_df['Churn'].mean()*100:.1f}%**")

    st.markdown('<div class="section-header">Churn by Feature</div>', unsafe_allow_html=True)

    categorical_features = [
        "SeniorCitizen", "Partner", "Dependents",
        "PhoneService", "PaperlessBilling",
        "Contract", "InternetService", "PaymentMethod",
    ]
    available = [f for f in categorical_features if f in raw_df.columns]
    selected_feature = st.selectbox("Select a feature to analyse churn rate:", available)

    if selected_feature:
        fig = churn_by_feature(raw_df, selected_feature)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-header">Numeric Feature Distributions</div>', unsafe_allow_html=True)
    st.plotly_chart(numeric_distributions(raw_df), use_container_width=True)

    st.markdown('<div class="section-header">Monthly Charges Distribution</div>', unsafe_allow_html=True)
    st.plotly_chart(monthly_charges_boxplot(raw_df), use_container_width=True)

    st.markdown('<div class="section-header">Feature Correlation Matrix</div>', unsafe_allow_html=True)
    with st.expander("Show correlation heatmap (all numeric features)"):
        st.plotly_chart(correlation_heatmap(X), use_container_width=True)

    st.markdown('<div class="section-header">Raw Data Sample</div>', unsafe_allow_html=True)
    with st.expander("Show raw data (first 50 rows)"):
        st.dataframe(raw_df.head(50), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3: MODEL RESULTS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🤖 Model Results":
    st.title("Model Performance")

    if not models_exist:
        st.warning("No trained models found. Run `python src/train.py` first.")
        st.stop()

    models = load_models()

    if not models:
        st.warning("Model files found but couldn't be loaded. Check models/ directory.")
        st.stop()

    # Reconstruct results dict from saved models
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    results = {}
    for name, pipeline in models.items():
        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1]
        from sklearn.metrics import (accuracy_score, precision_score,
                                      recall_score, f1_score, roc_auc_score,
                                      average_precision_score)
        results[name] = {
            "metrics": {
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred, zero_division=0),
                "recall": recall_score(y_test, y_pred, zero_division=0),
                "f1": f1_score(y_test, y_pred, zero_division=0),
                "roc_auc": roc_auc_score(y_test, y_prob),
                "pr_auc": average_precision_score(y_test, y_prob),
            },
            "y_pred": y_pred,
            "y_prob": y_prob,
            "y_test": y_test,
            "X_test": X_test,
        }

    st.markdown('<div class="section-header">Metrics Comparison</div>', unsafe_allow_html=True)
    metrics_df = model_comparison_table(results)
    st.dataframe(
        metrics_df.style.highlight_max(axis=0, color="#EAF3DE")
                        .format("{:.4f}"),
        use_container_width=True,
    )

    st.markdown('<div class="section-header">ROC and PR Curves</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(roc_curve_comparison(results), use_container_width=True)
    with col2:
        st.plotly_chart(pr_curve_comparison(results), use_container_width=True)

    st.markdown('<div class="section-header">Metric Radar</div>', unsafe_allow_html=True)
    st.plotly_chart(metrics_radar(results), use_container_width=True)

    st.markdown('<div class="section-header">Confusion Matrix</div>', unsafe_allow_html=True)
    selected_model = st.selectbox("Select model:", list(results.keys()))
    r = results[selected_model]
    st.plotly_chart(
        confusion_matrix_chart(r["y_test"], r["y_pred"], selected_model),
        use_container_width=True,
    )

    st.markdown('<div class="section-header">Business Impact Analysis</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        customer_value = st.slider("Avg annual customer value ($)", 500, 5000, 1200, 100)
    with col2:
        intervention_cost = st.slider("Cost per intervention ($)", 10, 200, 50, 10)

    best_model_name = max(results, key=lambda k: results[k]["metrics"]["roc_auc"])
    fig, best_thresh = business_impact_chart(
        results[best_model_name]["y_test"],
        results[best_model_name]["y_prob"],
        avg_customer_value=customer_value,
        cost_per_intervention=intervention_cost,
    )
    st.plotly_chart(fig, use_container_width=True)
    st.info(f"Optimal threshold: **{best_thresh:.2f}** — set your prediction cutoff here for maximum net revenue recovery.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4: SHAP EXPLAINABILITY
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🧠 SHAP Explainability":
    st.title("Model Explainability (SHAP)")
    st.markdown(
        "SHAP (SHapley Additive exPlanations) shows **which features drive each prediction** "
        "and **by how much** — making the model auditable and actionable."
    )

    if not models_exist:
        st.warning("No trained models found. Run `python src/train.py` first.")
        st.stop()

    best_pipeline = joblib.load(models_dir / "best_model.pkl")

    # Use a sample for speed (SHAP on full dataset takes ~2 min)
    X_sample = X.sample(min(500, len(X)), random_state=42)

    with st.spinner("Computing SHAP values..."):
        explainer, shap_values, X_transformed = load_shap(best_pipeline, X_sample)

    st.markdown('<div class="section-header">Global Feature Importance</div>', unsafe_allow_html=True)
    st.markdown("Mean absolute SHAP value — how much each feature shifts the prediction on average.")
    top_n = st.slider("Number of features to show:", 5, 20, 15)
    st.plotly_chart(global_importance_chart(shap_values, X_transformed, top_n=top_n),
                    use_container_width=True)

    st.markdown('<div class="section-header">SHAP Beeswarm — Direction and Magnitude</div>',
                unsafe_allow_html=True)
    st.markdown("Each dot is one customer. Red = high feature value. Position = impact on churn probability.")
    st.plotly_chart(beeswarm_data(shap_values, X_transformed, top_n=12),
                    use_container_width=True)

    st.markdown('<div class="section-header">Single Customer Explanation</div>',
                unsafe_allow_html=True)
    sample_idx = st.slider("Select customer index:", 0, len(X_sample) - 1, 0)
    pred_prob = best_pipeline.predict_proba(X_sample.iloc[[sample_idx]])[0][1]
    pred_label = "🔴 Likely to Churn" if pred_prob > 0.5 else "🟢 Likely to Stay"

    st.metric("Predicted Churn Probability", f"{pred_prob:.1%}", delta=pred_label)
    st.plotly_chart(
        single_prediction_waterfall(explainer, shap_values, X_transformed, sample_idx),
        use_container_width=True,
    )

    st.markdown('<div class="section-header">Feature Dependence Plot</div>',
                unsafe_allow_html=True)
    st.markdown("How does a single feature's value affect churn probability across all customers?")
    top_features = (
        pd.Series(np.abs(shap_values).mean(axis=0), index=X_transformed.columns)
        .nlargest(15).index.tolist()
    )
    dep_feature = st.selectbox("Select feature:", top_features)
    from explain import dependence_plot
    st.plotly_chart(dependence_plot(shap_values, X_transformed, dep_feature),
                    use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5: REAL-TIME PREDICTION
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🎯 Predict":
    st.title("Real-Time Churn Prediction")
    st.markdown("Enter a customer's details to get an instant churn probability with explanation.")

    if not models_exist:
        st.warning("No trained models found. Run `python src/train.py` first.")
        st.stop()

    best_pipeline = joblib.load(models_dir / "best_model.pkl")
    meta = joblib.load(models_dir / "metadata.pkl")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Account Info**")
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        contract = st.selectbox("Contract type", ["Month-to-month", "One year", "Two year"])
        paperless = st.selectbox("Paperless billing", ["Yes", "No"])

    with col2:
        st.markdown("**Services**")
        internet = st.selectbox("Internet service", ["Fiber optic", "DSL", "No"])
        online_security = st.selectbox("Online security", ["Yes", "No", "No internet service"])
        tech_support = st.selectbox("Tech support", ["Yes", "No", "No internet service"])

    with col3:
        st.markdown("**Billing**")
        monthly_charges = st.slider("Monthly charges ($)", 18.0, 120.0, 65.0, 0.5)
        payment = st.selectbox("Payment method", [
            "Electronic check", "Mailed check",
            "Bank transfer (automatic)", "Credit card (automatic)"
        ])
        senior = st.selectbox("Senior citizen", ["No", "Yes"])

    predict_btn = st.button("🔮 Predict Churn Probability", type="primary")

    if predict_btn:
        # Build a single-row DataFrame matching the training schema
        input_data = {
            "SeniorCitizen": 1 if senior == "Yes" else 0,
            "Partner": 0, "Dependents": 0,
            "tenure": tenure,
            "PhoneService": 1, "MultipleLines": 0,
            "OnlineSecurity": 1 if online_security == "Yes" else 0,
            "OnlineBackup": 0, "DeviceProtection": 0,
            "TechSupport": 1 if tech_support == "Yes" else 0,
            "StreamingTV": 0, "StreamingMovies": 0,
            "PaperlessBilling": 1 if paperless == "Yes" else 0,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": monthly_charges * max(tenure, 1),
            "gender": 1,
        }

        # One-hot encode internet service
        for svc in ["DSL", "Fiber optic", "No"]:
            input_data[f"InternetService_{svc}"] = 1 if internet == svc else 0

        # One-hot encode contract
        for c in ["Month-to-month", "One year", "Two year"]:
            input_data[f"Contract_{c}"] = 1 if contract == c else 0

        # One-hot encode payment
        for p in ["Bank transfer (automatic)", "Credit card (automatic)",
                   "Electronic check", "Mailed check"]:
            input_data[f"PaymentMethod_{p}"] = 1 if payment == p else 0

        input_df = pd.DataFrame([input_data])

        # Add engineered features
        input_df["charges_per_tenure"] = (
            input_df["TotalCharges"] / input_df["tenure"]
            if tenure > 0 else monthly_charges
        )
        service_cols = ["OnlineSecurity", "OnlineBackup", "DeviceProtection",
                         "TechSupport", "StreamingTV", "StreamingMovies"]
        input_df["service_count"] = input_df[service_cols].sum(axis=1)
        contract_yearly = [c for c in input_df.columns if "One year" in c or "Two year" in c]
        input_df["long_term_contract"] = input_df[contract_yearly].max(axis=1) if contract_yearly else 0
        input_df["monthly_to_total_ratio"] = (
            monthly_charges / (monthly_charges * max(tenure, 1))
        )
        input_df["tenure_band"] = min(int(tenure / 12), 3)

        # Align columns with training data
        for col in meta["feature_names"]:
            if col not in input_df.columns:
                input_df[col] = 0
        input_df = input_df[meta["feature_names"]]

        prob = best_pipeline.predict_proba(input_df)[0][1]
        pred = prob > 0.5

        st.divider()
        result_col1, result_col2, result_col3 = st.columns(3)

        with result_col1:
            color = "#E24B4A" if pred else "#1baf7a"
            st.markdown(f"""
            <div style="background:{color}15;border:2px solid {color};border-radius:12px;
                        padding:24px;text-align:center;">
                <div style="font-size:36px;font-weight:700;color:{color}">
                    {prob:.1%}
                </div>
                <div style="font-size:14px;color:{color};margin-top:6px">
                    {'🔴 HIGH CHURN RISK' if pred else '🟢 LOW CHURN RISK'}
                </div>
            </div>
            """, unsafe_allow_html=True)

        with result_col2:
            st.metric("Tenure", f"{tenure} months")
            st.metric("Monthly Charges", f"${monthly_charges:.2f}")

        with result_col3:
            st.metric("Contract", contract)
            st.metric("Annual Value at Risk", f"${monthly_charges * 12:,.0f}")

        if pred:
            st.markdown("""
            <div class="warning-box">
                <b>Recommended action:</b> Flag for retention campaign.
                Offer a discounted annual contract upgrade or loyalty discount.
                Target intervention within the next 30 days.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="insight-box">
                <b>Low risk customer.</b> Continue standard engagement.
                Consider upselling additional services to increase stickiness.
            </div>
            """, unsafe_allow_html=True)
