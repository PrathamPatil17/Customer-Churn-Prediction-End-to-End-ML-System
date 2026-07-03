"""
src/data_loader.py
------------------
Loads the IBM Telco Customer Churn dataset and applies
all preprocessing steps needed before modelling.
"""

import pandas as pd
import numpy as np
from pathlib import Path


RAW_PATH = Path("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")


def load_raw() -> pd.DataFrame:
    """Load raw CSV. Download from Kaggle if missing."""
    if not RAW_PATH.exists():
        raise FileNotFoundError(
            "Dataset not found. Download from:\n"
            "https://www.kaggle.com/datasets/blastchar/telco-customer-churn\n"
            f"and place it at: {RAW_PATH}"
        )
    df = pd.read_csv(RAW_PATH)
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Fix dtypes, handle missing values, drop useless columns."""
    df = df.copy()

    # TotalCharges is object due to whitespace in rows where tenure=0
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["MonthlyCharges"])

    # Drop customerID — not a feature
    df.drop(columns=["customerID"], inplace=True, errors="ignore")

    # Target to binary int
    df["Churn"] = (df["Churn"] == "Yes").astype(int)

    return df


def encode(df: pd.DataFrame) -> pd.DataFrame:
    """Binary encode yes/no columns; one-hot encode multi-class cols."""
    df = df.copy()

    binary_cols = [
        "gender", "Partner", "Dependents", "PhoneService",
        "PaperlessBilling", "MultipleLines",
        "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies",
    ]
    for col in binary_cols:
        if col in df.columns:
            df[col] = df[col].map(
                {"Yes": 1, "No": 0, "Male": 1, "Female": 0,
                 "No phone service": 0, "No internet service": 0}
            ).fillna(0).astype(int)

    multi_cols = ["InternetService", "Contract", "PaymentMethod"]
    df = pd.get_dummies(df, columns=multi_cols, drop_first=False)

    # Ensure bool cols become int
    bool_cols = df.select_dtypes(include="bool").columns
    df[bool_cols] = df[bool_cols].astype(int)

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create domain-driven features that capture business signals.
    These matter more than raw columns for churn prediction.
    """
    df = df.copy()

    # Charges per month relative to tenure — customers paying more
    # per month early in their lifecycle churn faster
    df["charges_per_tenure"] = np.where(
        df["tenure"] > 0,
        df["TotalCharges"] / df["tenure"],
        df["MonthlyCharges"],
    )

    # Service adoption rate — how many add-on services does customer use
    service_cols = [
        "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies",
    ]
    available = [c for c in service_cols if c in df.columns]
    df["service_count"] = df[available].sum(axis=1)

    # Is customer on a long-term contract? (strong churn signal)
    contract_yearly = [c for c in df.columns if "Contract_One" in c or "Contract_Two" in c]
    if contract_yearly:
        df["long_term_contract"] = df[contract_yearly].max(axis=1)
    else:
        df["long_term_contract"] = 0

    # Ratio of monthly charges to total charges (high ratio = newer customer)
    df["monthly_to_total_ratio"] = np.where(
        df["TotalCharges"] > 0,
        df["MonthlyCharges"] / df["TotalCharges"],
        1.0,
    )

    # Tenure band — segment customers by lifecycle stage
    df["tenure_band"] = pd.cut(
        df["tenure"],
        bins=[0, 12, 24, 48, 72],
        labels=[0, 1, 2, 3],
        include_lowest=True,
    ).astype(int)

    return df


def get_feature_target(df: pd.DataFrame):
    """Split into X, y and return feature names."""
    target = "Churn"
    X = df.drop(columns=[target])
    y = df[target]
    return X, y


def load_pipeline() -> tuple:
    """
    Full pipeline: raw → clean → encode → engineer → X, y.
    Returns (X, y, feature_names).
    """
    df = load_raw()
    df = clean(df)
    df = encode(df)
    df = engineer_features(df)
    X, y = get_feature_target(df)
    return X, y, list(X.columns)
