"""Data validation and cleaning for churn intelligence pipeline."""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from backend.app.config import GLOBAL_CONFIG

logger = logging.getLogger(__name__)

TARGET = GLOBAL_CONFIG.data.target
CATEGORICAL = list(GLOBAL_CONFIG.data.categorical_columns)
NUMERIC = list(GLOBAL_CONFIG.data.numeric_columns)
REQUIRED_COLUMNS = ["customerID", *CATEGORICAL, *NUMERIC, TARGET]

EXPECTED_CATEGORIES = {
    "gender": {"Female", "Male"},
    "Partner": {"Yes", "No"},
    "Dependents": {"Yes", "No"},
    "PhoneService": {"Yes", "No"},
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
    "PaymentMethod": {
        "Electronic check", "Mailed check", "Bank transfer (automatic)",
        "Credit card (automatic)",
    },
    TARGET: {"Yes", "No"},
}


def find_dataset(project_root: Path | None = None) -> Path:
    """Find customer dataset file under repository."""
    root = project_root or Path(__file__).resolve().parents[4]
    for candidate in (
        root / "churn_project" / "data" / "telco_churn.csv",
        root / "data" / "telco_churn.csv",
    ):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"telco_churn.csv not found under project root {root}")


def clean_dataset(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate, clean, and cast telco churn dataframe."""
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

    return df.reset_index(drop=True)
