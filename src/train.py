"""
src/train.py
------------
Trains Logistic Regression, Random Forest, XGBoost, and LightGBM.
Every experiment is tracked in MLflow.
Handles class imbalance with SMOTE.
Saves the best model to models/best_model.pkl
"""

import os
import joblib
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

import mlflow
import mlflow.sklearn
import mlflow.xgboost

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, average_precision_score,
    classification_report, confusion_matrix,
)
from xgboost import XGBClassifier
try:
    from lightgbm import LGBMClassifier
except (ImportError, OSError):
    LGBMClassifier = None
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

warnings.filterwarnings("ignore")

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.2


def get_models() -> dict:
    """
    Return all candidate models.
    Hyperparameters are tuned for this dataset — not defaults.
    """
    models = {
        "LogisticRegression": LogisticRegression(
            C=0.1,
            max_iter=1000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=3,  # handles imbalance
            eval_metric="logloss",
            use_label_encoder=False,
            random_state=RANDOM_STATE,
        ),
    }
    if LGBMClassifier is not None:
        models["LightGBM"] = LGBMClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            verbose=-1,
        )
    else:
        print("LightGBM unavailable on this machine (missing libomp) — skipping.")
    return models


def compute_metrics(y_true, y_pred, y_prob) -> dict:
    """Compute all evaluation metrics in one place."""
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y_true, y_pred, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y_true, y_prob), 4),
        "pr_auc": round(average_precision_score(y_true, y_prob), 4),
    }


def build_pipeline(model, use_smote: bool = True):
    """
    Wrap model in a preprocessing + optional SMOTE pipeline.
    SMOTE is only applied during training, not at inference.
    """
    steps = []

    # Standard scaling for LR; tree-based models don't need it
    # but it never hurts and keeps the pipeline uniform
    steps.append(("scaler", StandardScaler()))

    if use_smote:
        steps.append(("smote", SMOTE(random_state=RANDOM_STATE, k_neighbors=5)))
        steps.append(("model", model))
        return ImbPipeline(steps)
    else:
        steps.append(("model", model))
        return Pipeline(steps)


def train_all(X, y, experiment_name: str = "churn_prediction") -> dict:
    """
    Train all models, log everything to MLflow, return results dict.

    Returns:
        results: {model_name: {metrics, pipeline, y_pred, y_prob}}
    """
    mlflow.set_experiment(experiment_name)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE,
        stratify=y, random_state=RANDOM_STATE
    )

    models = get_models()
    results = {}
    best_auc = 0.0
    best_model_name = None

    print(f"\n{'='*60}")
    print(f"Training {len(models)} models on {len(X_train):,} samples")
    print(f"Test set: {len(X_test):,} samples | Churn rate: {y_test.mean():.1%}")
    print(f"{'='*60}\n")

    for name, model in models.items():
        print(f"Training {name}...")

        with mlflow.start_run(run_name=name):

            # Log hyperparameters
            mlflow.log_params(model.get_params())
            mlflow.log_param("smote", True)
            mlflow.log_param("test_size", TEST_SIZE)
            mlflow.log_param("train_samples", len(X_train))
            mlflow.log_param("test_samples", len(X_test))

            # Build pipeline and train
            pipeline = build_pipeline(model, use_smote=True)
            pipeline.fit(X_train, y_train)

            # Predictions
            y_pred = pipeline.predict(X_test)
            y_prob = pipeline.predict_proba(X_test)[:, 1]

            # Metrics
            metrics = compute_metrics(y_test, y_pred, y_prob)

            # CV score for stability check
            cv_scores = cross_val_score(
                pipeline, X_train, y_train,
                cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
                scoring="roc_auc", n_jobs=-1,
            )
            metrics["cv_roc_auc_mean"] = round(cv_scores.mean(), 4)
            metrics["cv_roc_auc_std"] = round(cv_scores.std(), 4)

            # Log metrics
            mlflow.log_metrics(metrics)

            # Log model
            if "XGBoost" in name:
                mlflow.xgboost.log_model(pipeline.named_steps["model"], name)
            else:
                mlflow.sklearn.log_model(pipeline, name)

            # Save pipeline locally too
            model_path = MODELS_DIR / f"{name.lower()}_pipeline.pkl"
            joblib.dump(pipeline, model_path)
            mlflow.log_artifact(str(model_path))

            results[name] = {
                "metrics": metrics,
                "pipeline": pipeline,
                "y_pred": y_pred,
                "y_prob": y_prob,
                "y_test": y_test,
                "X_test": X_test,
            }

            print(f"  ROC-AUC: {metrics['roc_auc']:.4f} | "
                  f"F1: {metrics['f1']:.4f} | "
                  f"CV AUC: {metrics['cv_roc_auc_mean']:.4f} ± {metrics['cv_roc_auc_std']:.4f}")

            # Track best model
            if metrics["roc_auc"] > best_auc:
                best_auc = metrics["roc_auc"]
                best_model_name = name

    # Save best model separately
    best_pipeline = results[best_model_name]["pipeline"]
    best_path = MODELS_DIR / "best_model.pkl"
    joblib.dump(best_pipeline, best_path)

    # Save metadata for dashboard
    meta = {
        "best_model": best_model_name,
        "best_roc_auc": best_auc,
        "feature_names": list(X.columns),
        "train_size": len(X_train),
        "test_size": len(X_test),
    }
    joblib.dump(meta, MODELS_DIR / "metadata.pkl")

    print(f"\n{'='*60}")
    print(f"Best model: {best_model_name} (ROC-AUC: {best_auc:.4f})")
    print(f"Saved to: {best_path}")
    print(f"MLflow UI: run 'mlflow ui' in this directory")
    print(f"{'='*60}\n")

    return results, X_test, y_test


def get_metrics_table(results: dict) -> pd.DataFrame:
    """Convert results dict to a clean comparison DataFrame."""
    rows = []
    for name, r in results.items():
        row = {"Model": name}
        row.update(r["metrics"])
        rows.append(row)
    df = pd.DataFrame(rows).set_index("Model")
    return df.sort_values("roc_auc", ascending=False)


if __name__ == "__main__":
    from data_loader import load_pipeline
    X, y, feature_names = load_pipeline()
    results, X_test, y_test = train_all(X, y)
    print(get_metrics_table(results))
