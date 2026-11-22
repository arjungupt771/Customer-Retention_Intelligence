"""
Phase 9 Integration Tests: End-to-End System & Cross-Phase Pipeline Validation
==============================================================================
Tests the end-to-end integration of all 9 roadmap phases:
- Phase 1: Model persistence and prediction service
- Phase 2: Evaluation diagnostics (ROC, PR, Lift, Confusion Matrix)
- Phase 3: Risk distribution, retention action center, and CSV export
- Phase 4: SHAP explainability (Global & Customer level)
- Phase 5: Cost-sensitive threshold optimization and persistence
- Phase 6: Bootstrap statistical confidence intervals
- Phase 7: Data drift & prediction monitoring
- Phase 8: Modular ML pipeline execution and configuration
"""

import os
import json
import pytest
import pandas as pd
from fastapi.testclient import TestClient

from api import app
from backend.app.config import GLOBAL_CONFIG
from backend.app.ml.pipeline import ChurnTrainingPipeline
from backend.app.ml.data.validation import clean_dataset, find_dataset
from churn_project.python.persistence import load_model_artifact
from churn_project.python.prediction_service import PredictionService


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


def test_full_ml_pipeline_lifecycle(tmp_path):
    """
    Validates Phase 8 and Phase 1:
    End-to-end modular pipeline training, persistence, and loading.
    """
    csv_file = find_dataset()
    df = clean_dataset(pd.read_csv(csv_file))
    sample = df.groupby("Churn", group_keys=False).sample(n=50, random_state=42).reset_index(drop=True)
    sample_csv = tmp_path / "sample.csv"
    sample.to_csv(sample_csv, index=False)

    pipeline = ChurnTrainingPipeline()
    result = pipeline.run(dataset_path=sample_csv, artifact_dir=tmp_path)

    assert result["status"] == "success"
    assert "selected_model" in result
    assert "threshold" in result
    assert (tmp_path / "model.joblib").exists()
    assert (tmp_path / "metadata.json").exists()
    assert (tmp_path / "metrics.json").exists()

    # Verify standard production artifacts load cleanly
    loaded_pipe, metadata, saved_metrics = load_model_artifact()
    assert loaded_pipe is not None
    assert "model_name" in metadata
    assert "threshold" in metadata
    assert "models" in saved_metrics


def test_prediction_api_integration(client):
    """
    Validates Phase 1:
    Inference via /api/predict using the persisted pipeline.
    """
    payload = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 1,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 85.5,
        "TotalCharges": 85.5
    }
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert "churn_probability" in data
    assert "predicted_churn" in data
    assert "risk_category" in data
    assert "threshold" in data
    assert 0.0 <= data["churn_probability"] <= 1.0
    assert isinstance(data["predicted_churn"], bool)
    assert data["risk_category"] in ["High", "Medium", "Low"]


def test_diagnostics_and_evaluation_integration(client):
    """
    Validates Phase 2:
    Model evaluation diagnostics, confusion matrix, ROC/PR curves, and lift deciles.
    """
    response = client.get("/api/model-diagnostics?threshold=0.35")
    assert response.status_code == 200
    data = response.json()

    assert "confusion_matrix" in data
    assert "classification_report" in data
    assert "roc_curve" in data
    assert "pr_curve" in data
    assert "lift_chart" in data
    assert "deciles" in data["lift_chart"]
    assert "gain_pct" in data["lift_chart"]["deciles"][0]

    cm = data["confusion_matrix"]
    assert cm["true_positives"] + cm["true_negatives"] + cm["false_positives"] + cm["false_negatives"] > 0

    roc = data["roc_curve"]
    assert 0.5 <= roc["roc_auc"] <= 1.0
    assert len(roc["points"]) > 5

    lift = data["lift_chart"]
    assert len(lift["deciles"]) == 10
    # Top decile lift should outperform overall base churn rate
    assert lift["deciles"][0]["lift"] >= 1.0


def test_retention_action_center_integration(client):
    """
    Validates Phase 3:
    Risk distribution, action center summary, customer ranking, and CSV export.
    """
    # 1. Risk distribution
    resp_dist = client.get("/api/risk-distribution")
    assert resp_dist.status_code == 200
    dist_data = resp_dist.json()
    assert "high" in dist_data
    assert "medium" in dist_data
    assert "low" in dist_data
    assert dist_data["high"]["count"] + dist_data["medium"]["count"] + dist_data["low"]["count"] > 1000

    # 2. Action center summary
    resp_summary = client.get("/api/action-center/summary")
    assert resp_summary.status_code == 200
    summary = resp_summary.json()
    assert summary["high_risk_customers"] > 0
    assert summary["monthly_revenue_at_risk"] > 0

    # 3. Action center customer list
    resp_customers = client.get("/api/action-center/customers?risk_category=High&limit=10")
    assert resp_customers.status_code == 200
    cust_data = resp_customers.json()
    assert len(cust_data["customers"]) <= 10
    for cust in cust_data["customers"]:
        assert cust["risk_category"] == "High"
        assert cust["churn_probability"] >= 0.70

    # 4. Action center CSV export
    resp_csv = client.get("/api/action-center/export-csv?risk_category=High&limit=20")
    assert resp_csv.status_code == 200
    assert "text/csv" in resp_csv.headers["content-type"]
    assert "customerID" in resp_csv.text
    assert "churn_probability" in resp_csv.text


def test_shap_explainability_integration(client):
    """
    Validates Phase 4:
    SHAP global feature attributions and individual customer waterfall explanations.
    """
    # 1. Global explanations
    resp_global = client.get("/api/explain/global")
    assert resp_global.status_code == 200
    global_data = resp_global.json()
    assert "feature_importance" in global_data
    assert len(global_data["feature_importance"]) >= 5
    top_feature = global_data["feature_importance"][0]
    assert "feature" in top_feature
    assert "mean_abs_shap" in top_feature

    # 2. Local customer explanation
    # Get a valid customer ID first
    cust_list = client.get("/api/action-center/customers?limit=1").json()
    first_cust = cust_list["customers"][0]
    cust_id = first_cust.get("customerID") or first_cust.get("customer_id")
    assert cust_id is not None

    resp_local = client.get(f"/api/explain/{cust_id}")
    assert resp_local.status_code == 200
    local_data = resp_local.json()
    assert local_data["customer_id"] == cust_id
    assert "churn_probability" in local_data
    assert "top_risk_factors" in local_data
    assert "top_protective_factors" in local_data
    assert "feature_contributions" in local_data


def test_threshold_optimization_and_persistence_integration(client):
    """
    Validates Phase 5:
    Cost-sensitive threshold evaluation and live persistence.
    """
    resp_analysis = client.get("/api/threshold-analysis?missed_churner_cost=5&false_positive_cost=1&intervention_cost=10")
    assert resp_analysis.status_code == 200
    data = resp_analysis.json()

    assert "threshold_curve" in data
    assert "optimal_thresholds" in data
    opt = data["optimal_thresholds"]
    assert "min_cost" in opt
    assert "max_f1" in opt
    assert 0.05 <= opt["min_cost"]["threshold"] <= 0.95

    # Persist the optimal threshold
    best_th = opt["min_cost"]["threshold"]
    resp_set = client.post("/api/threshold/set", json={"threshold": best_th})
    assert resp_set.status_code == 200
    assert resp_set.json()["threshold"] == best_th

    # Ensure /api/predict respects persisted threshold
    pred_resp = client.post("/api/predict", json={
        "tenure": 12,
        "MonthlyCharges": 70.0,
        "Contract": "Month-to-month",
        "PaymentMethod": "Electronic check"
    })
    assert pred_resp.status_code == 200
    assert pred_resp.json()["threshold"] == best_th


def test_statistical_confidence_intervals_integration(client):
    """
    Validates Phase 6:
    Bootstrap resampling producing 95% confidence intervals.
    """
    resp = client.get("/api/model-confidence?n_bootstraps=50&confidence_level=0.95")
    assert resp.status_code == 200
    data = resp.json()

    metrics = data["metrics"]
    for m in ["roc_auc", "pr_auc", "f1_score", "precision", "recall"]:
        assert m in metrics
        item = metrics[m]
        assert "estimate" in item
        assert "confidence_interval" in item
        ci = item["confidence_interval"]
        assert ci["lower"] <= item["estimate"] <= ci["upper"]


def test_drift_monitoring_integration(client):
    """
    Validates Phase 7:
    Drift detection framework comparing inference against baseline reference.
    """
    resp = client.get("/api/monitoring/drift")
    assert resp.status_code == 200
    data = resp.json()

    assert "overall_status" in data
    assert "feature_drift" in data
    assert "prediction_drift" in data
    assert len(data["feature_drift"]) > 0

    tenure_drift = data["feature_drift"]["tenure"]
    assert "psi" in tenure_drift
    assert "severity" in tenure_drift
    assert "ks_statistic" in tenure_drift
