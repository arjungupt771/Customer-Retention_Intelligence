"""Inference prediction service."""
from __future__ import annotations

import logging
from typing import Any, Mapping

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from backend.app.config import GLOBAL_CONFIG
from backend.app.ml.data.validation import CATEGORICAL, EXPECTED_CATEGORIES, NUMERIC
from backend.app.ml.models.persistence import load_model_artifact

logger = logging.getLogger(__name__)

COLUMN_MAPPING = {
    "gender": "gender",
    "seniorcitizen": "SeniorCitizen",
    "senior_citizen": "SeniorCitizen",
    "partner": "Partner",
    "dependents": "Dependents",
    "tenure": "tenure",
    "phoneservice": "PhoneService",
    "phone_service": "PhoneService",
    "multiplelines": "MultipleLines",
    "multiple_lines": "MultipleLines",
    "internetservice": "InternetService",
    "internet_service": "InternetService",
    "onlinesecurity": "OnlineSecurity",
    "online_security": "OnlineSecurity",
    "onlinebackup": "OnlineBackup",
    "online_backup": "OnlineBackup",
    "deviceprotection": "DeviceProtection",
    "device_protection": "DeviceProtection",
    "techsupport": "TechSupport",
    "tech_support": "TechSupport",
    "streamingtv": "StreamingTV",
    "streaming_tv": "StreamingTV",
    "streamingmovies": "StreamingMovies",
    "streaming_movies": "StreamingMovies",
    "contract": "Contract",
    "paperlessbilling": "PaperlessBilling",
    "paperless_billing": "PaperlessBilling",
    "paymentmethod": "PaymentMethod",
    "payment_method": "PaymentMethod",
    "monthlycharges": "MonthlyCharges",
    "monthly_charges": "MonthlyCharges",
    "totalcharges": "TotalCharges",
    "total_charges": "TotalCharges",
}

DEFAULT_FEATURE_VALUES: dict[str, Any] = {
    "gender": "Male",
    "SeniorCitizen": 0,
    "Partner": "No",
    "Dependents": "No",
    "tenure": 1,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "DSL",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 50.0,
    "TotalCharges": 50.0,
}


def categorize_risk(
    probability: float,
    risk_config: dict[str, dict[str, float]] | None = None,
) -> str:
    """Categorize churn probability into High, Medium, or Low."""
    high_threshold = GLOBAL_CONFIG.risk.high_min_probability
    med_threshold = GLOBAL_CONFIG.risk.medium_min_probability

    if risk_config:
        high_threshold = float(risk_config.get("high", {}).get("min_probability", high_threshold))
        med_threshold = float(risk_config.get("medium", {}).get("min_probability", med_threshold))

    if probability >= high_threshold:
        return "High"
    if probability >= med_threshold:
        return "Medium"
    return "Low"


class PredictionService:
    """Production inference prediction service."""

    def __init__(
        self,
        pipeline: Pipeline | None = None,
        metadata: dict[str, Any] | None = None,
        metrics: dict[str, Any] | None = None,
        threshold: float | None = None,
    ) -> None:
        if pipeline is None:
            pipeline, metadata, metrics = load_model_artifact()

        self.pipeline: Pipeline = pipeline
        self.metadata: dict[str, Any] = metadata or {}
        self.metrics: dict[str, Any] = metrics or {}
        self.threshold: float = float(threshold if threshold is not None else self.metadata.get("threshold", 0.35))
        self.risk_categories: dict[str, dict[str, float]] = self.metadata.get(
            "risk_categories",
            {
                "high": {"min_probability": GLOBAL_CONFIG.risk.high_min_probability},
                "medium": {"min_probability": GLOBAL_CONFIG.risk.medium_min_probability},
                "low": {"min_probability": GLOBAL_CONFIG.risk.low_min_probability},
            },
        )

    def set_threshold(self, new_threshold: float) -> None:
        if not 0.0 <= new_threshold <= 1.0:
            raise ValueError(f"Threshold must be between 0.0 and 1.0, got {new_threshold}")
        self.threshold = float(new_threshold)
        self.metadata["threshold"] = self.threshold

    def validate_and_format_input(self, raw_input: Mapping[str, Any]) -> pd.DataFrame:
        if not isinstance(raw_input, Mapping):
            raise ValueError("Input data must be a dictionary or mapping")

        normalized: dict[str, Any] = {}
        for k, v in raw_input.items():
            clean_key = str(k).strip()
            lookup_key = clean_key.lower().replace("-", "_").replace(" ", "_")
            canonical_name = COLUMN_MAPPING.get(lookup_key, clean_key)
            normalized[canonical_name] = v

        all_features = CATEGORICAL + NUMERIC
        row: dict[str, Any] = {}
        for col in all_features:
            if col in normalized and normalized[col] is not None:
                val = normalized[col]
            elif col == "TotalCharges":
                tenure_val = float(normalized.get("tenure", DEFAULT_FEATURE_VALUES["tenure"]))
                monthly_val = float(normalized.get("MonthlyCharges", DEFAULT_FEATURE_VALUES["MonthlyCharges"]))
                val = tenure_val * monthly_val
            else:
                val = DEFAULT_FEATURE_VALUES[col]
            row[col] = val

        for col in CATEGORICAL:
            val_str = str(row[col]).strip()
            allowed = EXPECTED_CATEGORIES.get(col, set())
            if allowed and val_str not in allowed:
                raise ValueError(
                    f"Invalid value '{row[col]}' for categorical feature '{col}'. Allowed: {sorted(allowed)}"
                )
            row[col] = val_str

        for col in NUMERIC:
            try:
                num_val = float(row[col])
            except (ValueError, TypeError) as exc:
                raise ValueError(f"Numeric feature '{col}' must be a valid number, got '{row[col]}'") from exc

            if not np.isfinite(num_val):
                raise ValueError(f"Numeric feature '{col}' must be finite")

            if col == "SeniorCitizen":
                if int(num_val) not in (0, 1):
                    raise ValueError(f"SeniorCitizen must be 0 or 1, got {num_val}")
                row[col] = int(num_val)
            elif col == "tenure":
                if num_val < 0 or num_val > 100:
                    raise ValueError(f"tenure must be between 0 and 100, got {num_val}")
                row[col] = int(num_val)
            elif col in ("MonthlyCharges", "TotalCharges"):
                if num_val < 0:
                    raise ValueError(f"{col} cannot be negative, got {num_val}")
                row[col] = round(num_val, 2)

        return pd.DataFrame([row], columns=all_features)

    def predict_one(self, customer_data: Mapping[str, Any]) -> dict[str, Any]:
        df = self.validate_and_format_input(customer_data)
        probabilities = self.pipeline.predict_proba(df)[0]
        churn_prob = float(probabilities[1])
        predicted_churn = bool(churn_prob >= self.threshold)
        risk = categorize_risk(churn_prob, self.risk_categories)

        return {
            "churn_probability": round(churn_prob, 2),
            "predicted_churn": predicted_churn,
            "risk_category": risk,
            "threshold": round(self.threshold, 2),
        }

    def predict_batch(self, df_or_records: list[dict[str, Any]] | pd.DataFrame) -> list[dict[str, Any]]:
        if isinstance(df_or_records, pd.DataFrame):
            records = df_or_records.to_dict(orient="records")
        else:
            records = list(df_or_records)
        return [self.predict_one(r) for r in records]
