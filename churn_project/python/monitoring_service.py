"""Data drift and model prediction monitoring service using PSI and Kolmogorov-Smirnov tests."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from churn_project.python.churn_analysis import CATEGORICAL, NUMERIC

logger = logging.getLogger(__name__)

DEFAULT_MONITORING_DIR = Path(__file__).resolve().parents[2] / "artifacts" / "monitoring"

DEFAULT_DRIFT_THRESHOLDS = {
    "low": 0.10,        # PSI < 0.10: Stable / No significant drift
    "moderate": 0.20,   # 0.10 <= PSI < 0.20: Moderate drift / Monitor closely
    "high": 0.30,       # PSI >= 0.20: High drift / Retraining or investigation warranted
}


def calculate_psi(
    expected_proportions: np.ndarray | list[float],
    actual_proportions: np.ndarray | list[float],
    epsilon: float = 1e-4,
) -> float:
    """Calculate Population Stability Index (PSI) between two discrete distributions."""
    exp = np.asarray(expected_proportions, dtype=float) + epsilon
    act = np.asarray(actual_proportions, dtype=float) + epsilon

    # Normalize to ensure sums equal 1
    exp /= np.sum(exp)
    act /= np.sum(act)

    psi_val = np.sum((act - exp) * np.log(act / exp))
    return float(np.round(np.maximum(0.0, psi_val), 4))


def classify_drift_severity(psi: float, thresholds: dict[str, float] | None = None) -> str:
    """Categorize PSI value into Low (Stable), Moderate, or High severity."""
    th = thresholds or DEFAULT_DRIFT_THRESHOLDS
    mod_th = th.get("low", 0.10)
    high_th = th.get("moderate", 0.20)

    if psi < mod_th:
        return "Low (Stable)"
    if psi < high_th:
        return "Moderate Drift"
    return "High Drift"


def compute_numerical_feature_drift(
    reference: pd.Series,
    current: pd.Series,
    n_bins: int = 10,
    thresholds: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Calculate numerical feature drift using PSI and 2-sample Kolmogorov-Smirnov test."""
    ref_clean = reference.dropna().to_numpy(dtype=float)
    curr_clean = current.dropna().to_numpy(dtype=float)

    if len(ref_clean) == 0 or len(curr_clean) == 0:
        return {"psi": 0.0, "severity": "Low (Stable)", "ks_statistic": 0.0, "ks_p_value": 1.0}

    # KS 2-sample test
    ks_res = stats.ks_2samp(ref_clean, curr_clean)

    # Binning based on reference quantiles
    quantiles = np.linspace(0, 1, n_bins + 1)
    bin_edges = np.percentile(ref_clean, quantiles * 100)
    bin_edges = np.unique(bin_edges)  # handle tied quantiles

    if len(bin_edges) < 2:
        bin_edges = np.array([ref_clean.min() - 1e-4, ref_clean.max() + 1e-4])

    ref_counts, _ = np.histogram(ref_clean, bins=bin_edges)
    curr_counts, _ = np.histogram(curr_clean, bins=bin_edges)

    ref_props = ref_counts / len(ref_clean)
    curr_props = curr_counts / len(curr_clean)

    psi_val = calculate_psi(ref_props, curr_props)
    severity = classify_drift_severity(psi_val, thresholds)

    return {
        "psi": psi_val,
        "severity": severity,
        "ks_statistic": round(float(ks_res.statistic), 4),
        "ks_p_value": round(float(ks_res.pvalue), 4),
        "reference_mean": round(float(np.mean(ref_clean)), 2),
        "current_mean": round(float(np.mean(curr_clean)), 2),
        "reference_std": round(float(np.std(ref_clean)), 2),
        "current_std": round(float(np.std(curr_clean)), 2),
    }


def compute_categorical_feature_drift(
    reference: pd.Series,
    current: pd.Series,
    thresholds: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Calculate categorical feature drift using discrete PSI across category frequencies."""
    ref_vc = reference.astype(str).value_counts(normalize=True)
    curr_vc = current.astype(str).value_counts(normalize=True)

    all_categories = sorted(set(ref_vc.index).union(set(curr_vc.index)))
    ref_props = np.array([ref_vc.get(c, 0.0) for c in all_categories])
    curr_props = np.array([curr_vc.get(c, 0.0) for c in all_categories])

    psi_val = calculate_psi(ref_props, curr_props)
    severity = classify_drift_severity(psi_val, thresholds)

    return {
        "psi": psi_val,
        "severity": severity,
        "categories": all_categories,
        "reference_distribution": {c: round(float(p), 4) for c, p in zip(all_categories, ref_props)},
        "current_distribution": {c: round(float(p), 4) for c, p in zip(all_categories, curr_props)},
    }


def compute_prediction_drift(
    reference_probs: np.ndarray | list[float],
    current_probs: np.ndarray | list[float],
    threshold: float = 0.35,
    high_risk_threshold: float = 0.70,
    thresholds: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Monitor prediction probability distribution shift and target risk classification shift."""
    ref_p = np.asarray(reference_probs, dtype=float)
    curr_p = np.asarray(current_probs, dtype=float)

    # 10 bins between 0.0 and 1.0 for probability distributions
    bins = np.linspace(0.0, 1.0, 11)
    ref_counts, _ = np.histogram(ref_p, bins=bins)
    curr_counts, _ = np.histogram(curr_p, bins=bins)

    ref_props = ref_counts / len(ref_p) if len(ref_p) else np.ones(10) / 10
    curr_props = curr_counts / len(curr_p) if len(curr_p) else np.ones(10) / 10

    psi_val = calculate_psi(ref_props, curr_props)
    severity = classify_drift_severity(psi_val, thresholds)

    return {
        "psi": psi_val,
        "severity": severity,
        "reference_avg_probability": round(float(np.mean(ref_p)), 4) if len(ref_p) else 0.0,
        "current_avg_probability": round(float(np.mean(curr_p)), 4) if len(curr_p) else 0.0,
        "reference_predicted_churn_pct": round(float(np.mean(ref_p >= threshold) * 100), 2) if len(ref_p) else 0.0,
        "current_predicted_churn_pct": round(float(np.mean(curr_p >= threshold) * 100), 2) if len(curr_p) else 0.0,
        "reference_high_risk_pct": round(float(np.mean(ref_p >= high_risk_threshold) * 100), 2) if len(ref_p) else 0.0,
        "current_high_risk_pct": round(float(np.mean(curr_p >= high_risk_threshold) * 100), 2) if len(curr_p) else 0.0,
    }


class DriftMonitoringService:
    """End-to-end data drift and prediction monitoring service."""

    def __init__(
        self,
        reference_df: pd.DataFrame,
        reference_probs: np.ndarray | None = None,
        thresholds: dict[str, float] | None = None,
        storage_dir: Path | str | None = None,
    ) -> None:
        self.reference_df = reference_df
        self.reference_probs = reference_probs
        self.thresholds = thresholds or DEFAULT_DRIFT_THRESHOLDS
        self.storage_dir = Path(storage_dir) if storage_dir else DEFAULT_MONITORING_DIR
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.save_reference_profile()

    def save_reference_profile(self) -> Path:
        """Store reference dataset statistical summary."""
        profile: dict[str, Any] = {
            "total_records": len(self.reference_df),
            "numerical_features": {},
            "categorical_features": {},
        }
        for col in NUMERIC:
            s = self.reference_df[col].dropna()
            profile["numerical_features"][col] = {
                "mean": round(float(s.mean()), 2),
                "std": round(float(s.std()), 2),
                "min": round(float(s.min()), 2),
                "max": round(float(s.max()), 2),
                "median": round(float(s.median()), 2),
                "missing_rate": round(float(self.reference_df[col].isna().mean()), 4),
            }
        for col in CATEGORICAL:
            s = self.reference_df[col].dropna().astype(str)
            profile["categorical_features"][col] = {
                "frequencies": {str(k): int(v) for k, v in s.value_counts().items()},
                "proportions": {str(k): round(float(v), 4) for k, v in s.value_counts(normalize=True).items()},
                "missing_rate": round(float(self.reference_df[col].isna().mean()), 4),
            }

        profile_path = self.storage_dir / "reference_stats.json"
        with open(profile_path, "w", encoding="utf-8") as f:
            json.dump(profile, f, indent=2)
        return profile_path

    def check_drift(
        self,
        current_df: pd.DataFrame,
        current_probs: np.ndarray | None = None,
        operating_threshold: float = 0.35,
    ) -> dict[str, Any]:
        """Perform comprehensive drift detection across features and model predictions."""
        feature_drift: dict[str, Any] = {}
        drifted_count = 0

        # Numerical features
        for col in NUMERIC:
            res = compute_numerical_feature_drift(
                self.reference_df[col], current_df[col], thresholds=self.thresholds
            )
            feature_drift[col] = {"type": "numeric", **res}
            if res["psi"] >= self.thresholds["low"]:
                drifted_count += 1

        # Categorical features
        for col in CATEGORICAL:
            res = compute_categorical_feature_drift(
                self.reference_df[col], current_df[col], thresholds=self.thresholds
            )
            feature_drift[col] = {"type": "categorical", **res}
            if res["psi"] >= self.thresholds["low"]:
                drifted_count += 1

        # Overall dataset drift status
        total_features = len(NUMERIC) + len(CATEGORICAL)
        drift_percentage = round((drifted_count / total_features) * 100, 1)

        if drift_percentage >= 30.0:
            overall_status = "High Drift"
        elif drift_percentage >= 15.0:
            overall_status = "Moderate Drift"
        else:
            overall_status = "No Significant Drift"

        # Prediction drift if probabilities provided
        pred_drift: dict[str, Any] = {}
        if self.reference_probs is not None and current_probs is not None:
            pred_drift = compute_prediction_drift(
                self.reference_probs,
                current_probs,
                threshold=operating_threshold,
                thresholds=self.thresholds,
            )

        return {
            "overall_status": overall_status,
            "drifted_features_count": drifted_count,
            "total_features": total_features,
            "drifted_features_pct": drift_percentage,
            "drift_thresholds": self.thresholds,
            "feature_drift": feature_drift,
            "prediction_drift": pred_drift,
        }
