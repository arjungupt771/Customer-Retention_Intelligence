"""Tests for Phase 4: SHAP Model Explainability."""
import pandas as pd
import pytest
from fastapi.testclient import TestClient

import api
from churn_project.python.churn_analysis import clean_dataset, find_dataset
from churn_project.python.explainability_service import (
    ModelExplainabilityService,
    clean_feature_name,
    parse_feature_info,
)
from churn_project.python.persistence import get_or_create_model


@pytest.fixture(scope="module")
def dataset():
    return clean_dataset(pd.read_csv(find_dataset()))


@pytest.fixture(scope="module")
def explainer_service(dataset):
    pipeline, _, _ = get_or_create_model()
    return ModelExplainabilityService(pipeline=pipeline, background_df=dataset)


def test_feature_name_cleaning_and_parsing():
    assert clean_feature_name("categorical__Contract_Month-to-month") == "Contract_Month-to-month"
    assert clean_feature_name("numeric__tenure") == "tenure"

    orig, cat = parse_feature_info("categorical__Contract_Month-to-month")
    assert orig == "Contract"
    assert cat == "Month-to-month"

    orig_num, cat_num = parse_feature_info("numeric__MonthlyCharges")
    assert orig_num == "MonthlyCharges"
    assert cat_num is None


def test_individual_customer_shap_explanation(explainer_service, dataset):
    customer = dataset.iloc[0]
    explanation = explainer_service.explain_instance(customer)

    assert "churn_probability" in explanation
    assert "predicted_churn" in explanation
    assert "base_value" in explanation
    assert "top_risk_factors" in explanation
    assert "top_protective_factors" in explanation
    assert "summary_text" in explanation
    assert "feature_contributions" in explanation

    # Verify probability is bounded
    assert 0.0 <= explanation["churn_probability"] <= 1.0

    # Risk factors must have positive contributions
    for rf in explanation["top_risk_factors"]:
        assert rf["contribution"] > 0
        assert "description" in rf

    # Protective factors must have negative contributions
    for pf in explanation["top_protective_factors"]:
        assert pf["contribution"] < 0
        assert "description" in pf


def test_global_shap_importance(explainer_service, dataset):
    global_imp = explainer_service.explain_global(dataset, max_samples=50)

    assert len(global_imp) > 0
    first = global_imp[0]
    assert "feature" in first
    assert "mean_abs_shap" in first
    assert "relative_importance_pct" in first
    assert first["mean_abs_shap"] >= 0

    # Ensure ranked in descending order
    importances = [item["mean_abs_shap"] for item in global_imp]
    assert importances == sorted(importances, reverse=True)


def test_api_explain_customer_endpoint(dataset):
    client = TestClient(api.app)
    valid_id = dataset.iloc[0]["customerID"]

    response = client.get(f"/api/explain/{valid_id}")
    assert response.status_code == 200
    data = response.json()

    assert data["customer_id"] == valid_id
    assert "churn_probability" in data
    assert "top_risk_factors" in data
    assert "top_protective_factors" in data
    assert "summary_text" in data


def test_api_explain_nonexistent_customer():
    client = TestClient(api.app)
    response = client.get("/api/explain/DOES_NOT_EXIST_9999")
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


def test_api_explain_global_endpoint():
    client = TestClient(api.app)
    response = client.get("/api/explain/global")
    assert response.status_code == 200
    data = response.json()
    assert "feature_importance" in data
    assert len(data["feature_importance"]) > 0
