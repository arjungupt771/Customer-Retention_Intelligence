"""Tests for Phase 7: Data Drift & Prediction Monitoring."""
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

import api
from churn_project.python.churn_analysis import clean_dataset, find_dataset
from churn_project.python.monitoring_service import (
    DriftMonitoringService,
    calculate_psi,
    classify_drift_severity,
    compute_categorical_feature_drift,
    compute_numerical_feature_drift,
    compute_prediction_drift,
)


@pytest.fixture(scope="module")
def dataset():
    return clean_dataset(pd.read_csv(find_dataset()))


def test_psi_identical_and_shifted_distributions():
    # Identical distributions should have PSI near 0
    p1 = np.array([0.2, 0.3, 0.5])
    p2 = np.array([0.2, 0.3, 0.5])
    assert calculate_psi(p1, p2) == 0.0

    # Shifted distributions should have positive PSI
    p_shifted = np.array([0.6, 0.3, 0.1])
    psi = calculate_psi(p1, p_shifted)
    assert psi > 0.20


def test_drift_severity_classification():
    assert classify_drift_severity(0.05) == "Low (Stable)"
    assert classify_drift_severity(0.15) == "Moderate Drift"
    assert classify_drift_severity(0.28) == "High Drift"


def test_numerical_feature_drift_calculation():
    np.random.seed(42)
    ref = pd.Series(np.random.normal(loc=50, scale=10, size=500))
    # Undrifted
    curr_same = pd.Series(np.random.normal(loc=50, scale=10, size=500))
    res_same = compute_numerical_feature_drift(ref, curr_same)
    assert res_same["psi"] < 0.10
    assert res_same["severity"] == "Low (Stable)"
    assert res_same["ks_p_value"] > 0.01

    # Shifted/Drifted
    curr_drifted = pd.Series(np.random.normal(loc=70, scale=15, size=500))
    res_drifted = compute_numerical_feature_drift(ref, curr_drifted)
    assert res_drifted["psi"] > 0.20
    assert res_drifted["severity"] == "High Drift"
    assert res_drifted["ks_statistic"] > 0.3


def test_categorical_feature_drift_calculation():
    ref = pd.Series(["Month-to-month"] * 60 + ["Two year"] * 40)
    curr_same = pd.Series(["Month-to-month"] * 59 + ["Two year"] * 41)
    res_same = compute_categorical_feature_drift(ref, curr_same)
    assert res_same["psi"] < 0.05

    curr_shifted = pd.Series(["Month-to-month"] * 10 + ["Two year"] * 90)
    res_shifted = compute_categorical_feature_drift(ref, curr_shifted)
    assert res_shifted["psi"] > 0.30
    assert res_shifted["severity"] == "High Drift"


def test_prediction_drift_monitoring():
    ref_probs = np.array([0.1, 0.2, 0.3, 0.8, 0.9])
    curr_probs = np.array([0.1, 0.2, 0.3, 0.75, 0.85])
    pred_res = compute_prediction_drift(ref_probs, curr_probs, threshold=0.5)

    assert "psi" in pred_res
    assert "severity" in pred_res
    assert "reference_avg_probability" in pred_res
    assert "current_avg_probability" in pred_res
    assert "reference_predicted_churn_pct" in pred_res
    assert "current_predicted_churn_pct" in pred_res


def test_end_to_end_drift_service(dataset, tmp_path):
    train_df = dataset.iloc[:500]
    test_df = dataset.iloc[500:1000]

    service = DriftMonitoringService(reference_df=train_df, storage_dir=tmp_path)
    report = service.check_drift(current_df=test_df)

    assert "overall_status" in report
    assert "drifted_features_count" in report
    assert "feature_drift" in report
    assert "MonthlyCharges" in report["feature_drift"]
    assert "Contract" in report["feature_drift"]
    assert (tmp_path / "reference_stats.json").exists()


def test_api_drift_monitoring_endpoint():
    client = TestClient(api.app)
    response = client.get("/api/monitoring/drift")
    assert response.status_code == 200
    data = response.json()

    assert "overall_status" in data
    assert "feature_drift" in data
    assert "prediction_drift" in data
    assert "drift_thresholds" in data
