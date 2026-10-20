"""Validated churn analysis and leakage-safe, reproducible model evaluation."""
from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import (StratifiedKFold, cross_validate,
                                     train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "Churn"
CATEGORICAL = ["gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
               "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
               "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
               "PaperlessBilling", "PaymentMethod"]
NUMERIC = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
REQUIRED_COLUMNS = ["customerID", *CATEGORICAL, *NUMERIC, TARGET]
EXPECTED_CATEGORIES = {
    "gender": {"Female", "Male"}, "Partner": {"Yes", "No"},
    "Dependents": {"Yes", "No"}, "PhoneService": {"Yes", "No"},
    "MultipleLines": {"Yes", "No", "No phone service"},
    "InternetService": {"DSL", "Fiber optic", "No"},
    "OnlineSecurity": {"Yes", "No", "No internet service"},
    "OnlineBackup": {"Yes", "No", "No internet service"},
    "DeviceProtection": {"Yes", "No", "No internet service"},
    "TechSupport": {"Yes", "No", "No internet service"},
    "StreamingTV": {"Yes", "No", "No internet service"},
    "StreamingMovies": {"Yes", "No", "No internet service"},
    "Contract": {"Month-to-month", "One year", "Two year"},
    "PaperlessBilling": {"Yes", "No"},
    "PaymentMethod": {"Electronic check", "Mailed check", "Bank transfer (automatic)",
                       "Credit card (automatic)"},
    TARGET: {"Yes", "No"},
}


def find_dataset(project_root: Path | None = None) -> Path:
    root = project_root or Path(__file__).resolve().parents[2]
    for candidate in (root / "churn_project" / "data" / "telco_churn.csv",
                      root / "data" / "telco_churn.csv"):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"telco_churn.csv not found under project root {root}")


def clean_dataset(frame: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(set(REQUIRED_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")
    unexpected_columns = sorted(set(frame.columns) - set(REQUIRED_COLUMNS))
    if unexpected_columns:
        raise ValueError(f"Unexpected columns: {', '.join(unexpected_columns)}")
    df = frame[REQUIRED_COLUMNS].copy()
    if df.empty:
        raise ValueError("Dataset must contain at least one customer")
    df["customerID"] = df.customerID.astype("string").str.strip()
    if df.customerID.isna().any() or df.customerID.eq("").any():
        raise ValueError("customerID cannot be empty")
    if df.customerID.duplicated().any():
        duplicate_count = int(df.customerID.duplicated().sum())
        raise ValueError(f"Found {duplicate_count} duplicate customerID values")
    for column, allowed in EXPECTED_CATEGORIES.items():
        df[column] = df[column].astype("string").str.strip()
        invalid = sorted(set(df[column].dropna().unique()) - allowed)
        if invalid:
            raise ValueError(f"Unexpected {column} values: {invalid}")
        if df[column].isna().any():
            raise ValueError(f"{column} cannot be missing")
    for column in NUMERIC:
        df[column] = pd.to_numeric(df[column], errors="coerce")
        if column == "TotalCharges":
            new_customer_blank = df.TotalCharges.isna() & df.tenure.eq(0)
            df.loc[new_customer_blank, "TotalCharges"] = 0
        if df[column].isna().any() or not np.isfinite(df[column]).all():
            raise ValueError(f"{column} must contain finite numeric values")
    if not df.SeniorCitizen.isin([0, 1]).all():
        raise ValueError("SeniorCitizen must contain only 0 or 1")
    if not df.tenure.between(0, 100).all():
        raise ValueError("tenure must be between 0 and 100 months")
    if (df[["MonthlyCharges", "TotalCharges"]] < 0).any().any():
        raise ValueError("Charges cannot be negative")
    # Blank TotalCharges in the source are valid for new customers; the published data uses 0.
    return df.reset_index(drop=True)


def key_metrics(df: pd.DataFrame) -> dict:
    total = int(len(df))
    churned_mask = df[TARGET].eq("Yes")
    active_mask = df[TARGET].eq("No")
    churned = int(churned_mask.sum())
    active = int(active_mask.sum())
    active_revenue = float(df.loc[active_mask, "MonthlyCharges"].sum())
    historical_revenue = float(df.loc[churned_mask, "MonthlyCharges"].sum())
    all_revenue = active_revenue + historical_revenue
    return {
        "total_customers": total, "active_customers": active, "churned_customers": churned,
        "churn_rate": round(churned / total * 100, 2) if total else 0.0,
        "active_monthly_revenue": round(active_revenue, 2),
        "historical_churned_monthly_revenue": round(historical_revenue, 2),
        # Excludes already-churned customers; descriptive exposure, not expected recovery.
        "active_monthly_revenue_at_risk": round(active_revenue, 2),
        "revenue_loss_percentage": round(historical_revenue / all_revenue * 100, 2) if all_revenue else 0.0,
        "population": "All customers in the supplied dataset", "period": "MonthlyCharges snapshot; no date field",
    }


def make_pipeline(model) -> Pipeline:
    preprocessor = ColumnTransformer([
        ("categorical", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), CATEGORICAL),
        ("numeric", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), NUMERIC),
    ])
    return Pipeline([("preprocess", preprocessor), ("model", model)])


def build_model_evaluation(df: pd.DataFrame) -> dict:
    X = df[CATEGORICAL + NUMERIC]
    y = df[TARGET].eq("Yes").astype(int)
    if y.nunique() != 2 or y.value_counts().min() < 2:
        raise ValueError("Model evaluation requires at least two rows in each target class")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y)
    baseline_probability = float(y_train.mean())
    baseline_prediction = np.full(len(y_test), int(baseline_probability >= 0.5))
    candidates = {
        "Logistic Regression": LogisticRegression(max_iter=1500, class_weight="balanced", random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=8,
                                                class_weight="balanced", random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=120, max_depth=2, random_state=42),
    }
    cv = StratifiedKFold(n_splits=min(5, int(y_train.value_counts().min())), shuffle=True, random_state=42)
    metrics = {"precision": "precision", "recall": "recall", "f1": "f1",
               "roc_auc": "roc_auc", "pr_auc": "average_precision"}
    evaluated, fitted = {}, {}
    for name, estimator in candidates.items():
        pipeline = make_pipeline(estimator)
        cv_result = cross_validate(pipeline, X_train, y_train, cv=cv, scoring=metrics, n_jobs=1)
        X_fit, X_validation, y_fit, y_validation = train_test_split(
            X_train, y_train, test_size=0.2, random_state=314, stratify=y_train)
        threshold_model = make_pipeline(estimator).fit(X_fit, y_fit)
        val_probs = threshold_model.predict_proba(X_validation)[:, 1]
        threshold = choose_threshold(y_validation.to_numpy(), val_probs)
        pipeline.fit(X_train, y_train)
        test_probs = pipeline.predict_proba(X_test)[:, 1]
        prediction = (test_probs >= threshold).astype(int)
        evaluated[name] = {
            "precision": round(float(precision_score(y_test, prediction, zero_division=0)), 3),
            "recall": round(float(recall_score(y_test, prediction, zero_division=0)), 3),
            "f1_score": round(float(f1_score(y_test, prediction, zero_division=0)), 3),
            "accuracy": round(float(accuracy_score(y_test, prediction)), 3),
            "auc_roc": round(float(roc_auc_score(y_test, test_probs)), 3),
            "pr_auc": round(float(average_precision_score(y_test, test_probs)), 3),
            "threshold": round(threshold, 3), "cv": {
                key: {"mean": round(float(np.mean(cv_result[f"test_{key}"])), 3),
                      "std": round(float(np.std(cv_result[f"test_{key}"])), 3)}
                for key in metrics
            },
            "test_population": int(len(y_test)), "test_churners": int(y_test.sum()),
        }
        fitted[name] = pipeline
    selected = max(evaluated, key=lambda name: evaluated[name]["cv"]["pr_auc"]["mean"])
    model = fitted[selected]
    try:
        names = model.named_steps["preprocess"].get_feature_names_out()
        estimator = model.named_steps["model"]
        weights = getattr(estimator, "feature_importances_", None)
        if weights is None and hasattr(estimator, "coef_"):
            weights = np.abs(estimator.coef_[0])
        importance = [] if weights is None else [
            {"feature": str(n).replace("categorical__", "").replace("numeric__", ""),
             "importance": round(float(w), 5),
             "interpretation": "association, not causation"}
            for n, w in sorted(zip(names, weights), key=lambda pair: pair[1], reverse=True)[:10]
        ]
    except (AttributeError, ValueError):
        importance = []
    baseline_probabilities = np.full(len(y_test), baseline_probability)
    baseline = {
        "precision": round(float(precision_score(y_test, baseline_prediction, zero_division=0)), 3),
        "recall": round(float(recall_score(y_test, baseline_prediction, zero_division=0)), 3),
        "f1_score": round(float(f1_score(y_test, baseline_prediction, zero_division=0)), 3),
        "accuracy": round(float(accuracy_score(y_test, baseline_prediction)), 3),
        "auc_roc": round(float(roc_auc_score(y_test, baseline_probabilities)), 3),
        "pr_auc": round(float(average_precision_score(y_test, baseline_probabilities)), 3),
    }
    return {"baseline": baseline, "models": evaluated, "selected_model": selected,
            "feature_importance": importance,
            "evaluation_note": "One stratified 25% held-out test set; threshold selected on training data only. CV preprocessing is fitted within each fold.",
            "threshold_cost_assumption": {"missed_churner_cost": 5, "false_positive_outreach_cost": 1,
                                         "description": "Illustrative relative costs; change to match operating economics."}}


def choose_threshold(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    # The costs are relative illustrative units, and selection uses training probabilities only.
    choices = np.linspace(0.05, 0.95, 91)
    return float(min(choices, key=lambda t: 5 * np.sum((y_true == 1) & (probabilities < t))
                     + np.sum((y_true == 0) & (probabilities >= t))))


def main() -> None:
    df = clean_dataset(pd.read_csv(find_dataset()))
    print(json.dumps({"data_validation": {"rows": len(df), "columns": len(df.columns),
          "duplicate_customer_ids": int(df.customerID.duplicated().sum()),
          "missing_values": int(df.isna().sum().sum())},
          "business_metrics": key_metrics(df), "machine_learning": build_model_evaluation(df)}, indent=2))


if __name__ == "__main__":
    main()
