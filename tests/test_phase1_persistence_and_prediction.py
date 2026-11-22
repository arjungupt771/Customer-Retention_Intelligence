"""Tests for Phase 1: Model persistence and prediction pipeline."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression

import api
from churn_project.python.churn_analysis import clean_dataset, find_dataset, make_pipeline
from churn_project.python.persistence import (
    load_model_artifact,
    save_model_artifact,
    get_or_create_model,
)
from churn_project.python.prediction_service import (
    PredictionService,
    categorize_risk,
)


@pytest.fixture
def sample_customer():
    return {
        "gender": "Male",
        "tenure": 12,
        "monthly_charges": 75.5,
        "contract": "Month-to-month",
        "payment_method": "Electronic check",
    }


def test_model_artifact_persistence(tmp_path):
    pipeline = make_pipeline(LogisticRegression())
    dummy_df = pd.DataFrame([
        {
            "gender": "Female", "Partner": "Yes", "Dependents": "No",
            "PhoneService": "Yes", "MultipleLines": "No", "InternetService": "DSL",
            "OnlineSecurity": "Yes", "OnlineBackup": "No", "DeviceProtection": "Yes",
            "TechSupport": "No", "StreamingTV": "No", "StreamingMovies": "No",
            "Contract": "Month-to-month", "PaperlessBilling": "Yes",
            "PaymentMethod": "Electronic check", "SeniorCitizen": 0, "tenure": 10,
            "MonthlyCharges": 50.0, "TotalCharges": 500.0,
        },
        {
            "gender": "Male", "Partner": "No", "Dependents": "No",
            "PhoneService": "Yes", "MultipleLines": "No", "InternetService": "DSL",
            "OnlineSecurity": "No", "OnlineBackup": "No", "DeviceProtection": "No",
            "TechSupport": "No", "StreamingTV": "No", "StreamingMovies": "No",
            "Contract": "One year", "PaperlessBilling": "No",
            "PaymentMethod": "Mailed check", "SeniorCitizen": 0, "tenure": 24,
            "MonthlyCharges": 20.0, "TotalCharges": 480.0,
        },
    ])
    pipeline.fit(dummy_df, np.array([1, 0]))

    metrics = {"precision": 0.8, "recall": 0.75, "f1_score": 0.77}
    saved_dir = save_model_artifact(
        pipeline=pipeline,
        model_name="Test Logistic Regression",
        metrics=metrics,
        threshold=0.45,
        artifact_dir=tmp_path,
        version="0.1.0",
    )

    assert (tmp_path / "model.joblib").exists()
    assert (tmp_path / "metadata.json").exists()
    assert (tmp_path / "metrics.json").exists()

    with open(tmp_path / "metadata.json", encoding="utf-8") as f:
        metadata = json.load(f)
    assert metadata["model_name"] == "Test Logistic Regression"
    assert metadata["threshold"] == 0.45
    assert metadata["version"] == "0.1.0"


def test_model_loading_system(tmp_path):
    pipeline = make_pipeline(LogisticRegression())
    dummy_df = pd.DataFrame([
        {
            "gender": "Male", "Partner": "No", "Dependents": "No",
            "PhoneService": "Yes", "MultipleLines": "No", "InternetService": "DSL",
            "OnlineSecurity": "No", "OnlineBackup": "No", "DeviceProtection": "No",
            "TechSupport": "No", "StreamingTV": "No", "StreamingMovies": "No",
            "Contract": "Month-to-month", "PaperlessBilling": "No",
            "PaymentMethod": "Mailed check", "SeniorCitizen": 0, "tenure": 5,
            "MonthlyCharges": 30.0, "TotalCharges": 150.0,
        },
        {
            "gender": "Female", "Partner": "Yes", "Dependents": "Yes",
            "PhoneService": "Yes", "MultipleLines": "Yes", "InternetService": "Fiber optic",
            "OnlineSecurity": "Yes", "OnlineBackup": "Yes", "DeviceProtection": "Yes",
            "TechSupport": "Yes", "StreamingTV": "Yes", "StreamingMovies": "Yes",
            "Contract": "Two year", "PaperlessBilling": "Yes",
            "PaymentMethod": "Bank transfer (automatic)", "SeniorCitizen": 0, "tenure": 40,
            "MonthlyCharges": 80.0, "TotalCharges": 3200.0,
        },
    ])
    pipeline.fit(dummy_df, np.array([0, 1]))

    save_model_artifact(pipeline, "Saved Model", {"auc": 0.85}, 0.35, artifact_dir=tmp_path)
    loaded_pipeline, meta, loaded_metrics = load_model_artifact(tmp_path)

    assert meta["model_name"] == "Saved Model"
    assert meta["threshold"] == 0.35
    assert loaded_metrics["auc"] == 0.85
    assert hasattr(loaded_pipeline, "predict_proba")


def test_model_loading_missing_artifact(tmp_path):
    empty_dir = tmp_path / "non_existent"
    with pytest.raises(FileNotFoundError):
        load_model_artifact(empty_dir)


def test_model_loading_corrupted_artifact(tmp_path):
    corrupted_dir = tmp_path / "corrupted"
    corrupted_dir.mkdir()
    (corrupted_dir / "model.joblib").write_text("invalid binary content", encoding="utf-8")
    (corrupted_dir / "metadata.json").write_text("{bad json", encoding="utf-8")
    (corrupted_dir / "metrics.json").write_text("{}", encoding="utf-8")

    with pytest.raises(RuntimeError):
        load_model_artifact(corrupted_dir)


def test_prediction_service_single_and_batch(sample_customer):
    service = PredictionService()
    res = service.predict_one(sample_customer)

    assert "churn_probability" in res
    assert "predicted_churn" in res
    assert "risk_category" in res
    assert "threshold" in res
    assert 0.0 <= res["churn_probability"] <= 1.0
    assert isinstance(res["predicted_churn"], bool)
    assert res["risk_category"] in ("High", "Medium", "Low")

    batch = service.predict_batch([sample_customer, sample_customer])
    assert len(batch) == 2


def test_risk_categorization_boundaries():
    assert categorize_risk(0.85) == "High"
    assert categorize_risk(0.70) == "High"
    assert categorize_risk(0.69) == "Medium"
    assert categorize_risk(0.30) == "Medium"
    assert categorize_risk(0.29) == "Low"
    assert categorize_risk(0.00) == "Low"


def test_prediction_api_success(sample_customer):
    client = TestClient(api.app)
    response = client.post("/api/predict", json=sample_customer)

    assert response.status_code == 200
    data = response.json()
    assert "churn_probability" in data
    assert "predicted_churn" in data
    assert "risk_category" in data
    assert "threshold" in data
    assert isinstance(data["predicted_churn"], bool)
    assert data["risk_category"] in ("High", "Medium", "Low")


def test_prediction_api_rejects_invalid_categorical():
    client = TestClient(api.app)
    invalid_data = {
        "gender": "Non-binary",  # Not in allowed categories
        "tenure": 12,
        "monthly_charges": 75.5,
        "contract": "Month-to-month",
    }
    response = client.post("/api/predict", json=invalid_data)
    assert response.status_code == 422
    assert "Invalid value 'Non-binary'" in response.json()["detail"]


def test_prediction_api_rejects_negative_numeric():
    client = TestClient(api.app)
    invalid_data = {
        "tenure": -5,
        "monthly_charges": 75.5,
    }
    response = client.post("/api/predict", json=invalid_data)
    assert response.status_code == 422
    assert "tenure must be between 0 and 100" in response.json()["detail"]


def test_prediction_api_rejects_malformed_input():
    client = TestClient(api.app)
    response = client.post(
        "/api/predict",
        content="not a json",
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 422
