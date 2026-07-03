"""
tests/test_pipeline.py
-----------------------
Unit tests for data loading, feature engineering, and model contracts.
Run with: pytest tests/ -v
"""

import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data_loader import clean, encode, engineer_features, get_feature_target


# ── Fixtures ──────────────────────────────────────────────────────────────────
@pytest.fixture
def sample_raw_df():
    """Minimal synthetic dataframe mimicking IBM Telco schema."""
    return pd.DataFrame({
        "customerID": ["1", "2", "3", "4", "5"],
        "gender": ["Male", "Female", "Male", "Female", "Male"],
        "SeniorCitizen": [0, 1, 0, 0, 1],
        "Partner": ["Yes", "No", "Yes", "No", "Yes"],
        "Dependents": ["No", "No", "Yes", "No", "No"],
        "tenure": [1, 34, 2, 45, 72],
        "PhoneService": ["No", "Yes", "Yes", "No", "Yes"],
        "MultipleLines": ["No phone service", "No", "No", "No phone service", "Yes"],
        "InternetService": ["DSL", "DSL", "DSL", "Fiber optic", "Fiber optic"],
        "OnlineSecurity": ["No", "Yes", "Yes", "No", "No"],
        "OnlineBackup": ["Yes", "No", "Yes", "No", "No"],
        "DeviceProtection": ["No", "Yes", "No", "Yes", "No"],
        "TechSupport": ["No", "No", "No", "Yes", "No"],
        "StreamingTV": ["No", "No", "No", "No", "Yes"],
        "StreamingMovies": ["No", "No", "No", "No", "No"],
        "Contract": ["Month-to-month", "One year", "Month-to-month", "One year", "Two year"],
        "PaperlessBilling": ["Yes", "No", "Yes", "No", "Yes"],
        "PaymentMethod": ["Electronic check", "Mailed check",
                           "Mailed check", "Bank transfer (automatic)",
                           "Credit card (automatic)"],
        "MonthlyCharges": [29.85, 56.95, 53.85, 42.30, 70.70],
        "TotalCharges": ["29.85", "1889.50", "108.15", "1840.75", "5091.85"],
        "Churn": ["No", "No", "Yes", "No", "No"],
    })


# ── Data cleaning tests ───────────────────────────────────────────────────────
class TestClean:
    def test_removes_customer_id(self, sample_raw_df):
        df = clean(sample_raw_df)
        assert "customerID" not in df.columns

    def test_total_charges_is_numeric(self, sample_raw_df):
        df = clean(sample_raw_df)
        assert df["TotalCharges"].dtype == float

    def test_churn_is_binary_int(self, sample_raw_df):
        df = clean(sample_raw_df)
        assert set(df["Churn"].unique()).issubset({0, 1})
        assert df["Churn"].dtype == int

    def test_no_missing_after_clean(self, sample_raw_df):
        df = clean(sample_raw_df)
        assert df.isnull().sum().sum() == 0

    def test_handles_whitespace_total_charges(self):
        """Edge case: TotalCharges with whitespace (tenure=0 rows)."""
        df = pd.DataFrame({
            "customerID": ["X"],
            "gender": ["Male"], "SeniorCitizen": [0],
            "Partner": ["No"], "Dependents": ["No"],
            "tenure": [0], "PhoneService": ["Yes"],
            "MultipleLines": ["No"], "InternetService": ["DSL"],
            "OnlineSecurity": ["No"], "OnlineBackup": ["No"],
            "DeviceProtection": ["No"], "TechSupport": ["No"],
            "StreamingTV": ["No"], "StreamingMovies": ["No"],
            "Contract": ["Month-to-month"], "PaperlessBilling": ["Yes"],
            "PaymentMethod": ["Electronic check"],
            "MonthlyCharges": [29.85], "TotalCharges": [" "],
            "Churn": ["No"],
        })
        result = clean(df)
        assert not pd.isna(result["TotalCharges"].iloc[0])


# ── Encoding tests ────────────────────────────────────────────────────────────
class TestEncode:
    def test_no_object_columns_remain(self, sample_raw_df):
        df = clean(sample_raw_df)
        df = encode(df)
        object_cols = df.select_dtypes(include="object").columns.tolist()
        assert len(object_cols) == 0, f"Object cols remain: {object_cols}"

    def test_one_hot_creates_expected_columns(self, sample_raw_df):
        df = clean(sample_raw_df)
        df = encode(df)
        assert any("Contract" in col for col in df.columns)
        assert any("InternetService" in col for col in df.columns)

    def test_binary_columns_are_0_or_1(self, sample_raw_df):
        df = clean(sample_raw_df)
        df = encode(df)
        for col in ["Partner", "Dependents", "PhoneService", "PaperlessBilling"]:
            if col in df.columns:
                assert set(df[col].unique()).issubset({0, 1}), f"{col} not binary"


# ── Feature engineering tests ─────────────────────────────────────────────────
class TestFeatureEngineering:
    def test_engineered_features_exist(self, sample_raw_df):
        df = clean(sample_raw_df)
        df = encode(df)
        df = engineer_features(df)
        expected = ["charges_per_tenure", "service_count",
                     "monthly_to_total_ratio", "tenure_band"]
        for feat in expected:
            assert feat in df.columns, f"Missing: {feat}"

    def test_service_count_range(self, sample_raw_df):
        df = clean(sample_raw_df)
        df = encode(df)
        df = engineer_features(df)
        assert df["service_count"].min() >= 0
        assert df["service_count"].max() <= 6

    def test_monthly_to_total_ratio_bounded(self, sample_raw_df):
        df = clean(sample_raw_df)
        df = encode(df)
        df = engineer_features(df)
        assert (df["monthly_to_total_ratio"] > 0).all()
        assert (df["monthly_to_total_ratio"] <= 1.01).all()

    def test_tenure_band_is_categorical(self, sample_raw_df):
        df = clean(sample_raw_df)
        df = encode(df)
        df = engineer_features(df)
        assert set(df["tenure_band"].unique()).issubset({0, 1, 2, 3})


# ── Feature-target split tests ────────────────────────────────────────────────
class TestFeatureTargetSplit:
    def test_churn_not_in_X(self, sample_raw_df):
        df = clean(sample_raw_df)
        df = encode(df)
        df = engineer_features(df)
        X, y = get_feature_target(df)
        assert "Churn" not in X.columns

    def test_y_is_binary(self, sample_raw_df):
        df = clean(sample_raw_df)
        df = encode(df)
        df = engineer_features(df)
        X, y = get_feature_target(df)
        assert set(y.unique()).issubset({0, 1})

    def test_X_y_same_length(self, sample_raw_df):
        df = clean(sample_raw_df)
        df = encode(df)
        df = engineer_features(df)
        X, y = get_feature_target(df)
        assert len(X) == len(y)

    def test_no_missing_in_X(self, sample_raw_df):
        df = clean(sample_raw_df)
        df = encode(df)
        df = engineer_features(df)
        X, y = get_feature_target(df)
        assert X.isnull().sum().sum() == 0
