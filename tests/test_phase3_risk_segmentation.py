"""Tests for Phase 3: Customer Risk Segmentation & Retention Action Center."""
import io
import pandas as pd
import pytest
from fastapi.testclient import TestClient

import api
from churn_project.python.churn_analysis import clean_dataset, find_dataset
from churn_project.python.prediction_service import PredictionService
from churn_project.python.risk_service import RiskSegmentationService


@pytest.fixture(scope="module")
def dataset():
    return clean_dataset(pd.read_csv(find_dataset()))


@pytest.fixture(scope="module")
def risk_service(dataset):
    pred_svc = PredictionService()
    return RiskSegmentationService(prediction_service=pred_svc, df=dataset)


def test_risk_distribution_aggregation(risk_service):
    dist = risk_service.get_risk_distribution()

    assert "high" in dist
    assert "medium" in dist
    assert "low" in dist

    for cat in ("high", "medium", "low"):
        assert "count" in dist[cat]
        assert "percentage" in dist[cat]
        assert dist[cat]["count"] >= 0
        assert 0.0 <= dist[cat]["percentage"] <= 100.0

    total_pct = sum(dist[c]["percentage"] for c in ("high", "medium", "low"))
    assert pytest.approx(total_pct, abs=0.5) == 100.0


def test_action_center_summary(risk_service):
    summary = risk_service.get_action_center_summary()

    assert "total_customers" in summary
    assert "high_risk_customers" in summary
    assert "high_risk_percentage" in summary
    assert "average_churn_probability" in summary
    assert "predicted_churn_rate" in summary
    assert "monthly_revenue_at_risk" in summary
    assert summary["total_customers"] > 0
    assert summary["monthly_revenue_at_risk"] >= 0


def test_customer_filtering_and_ranking(risk_service):
    # Test filtering by High risk
    high_risk_custs, total_high = risk_service.filter_customers(risk_category="High", limit=10)
    assert len(high_risk_custs) <= 10
    for c in high_risk_custs:
        assert c["risk_category"] == "High"
        assert c["churn_probability"] >= 0.70

    # Test filtering by contract
    m2m_custs, total_m2m = risk_service.filter_customers(contract="Month-to-month", limit=10)
    for c in m2m_custs:
        assert c["Contract"] == "Month-to-month"

    # Test filtering by tenure range
    tenure_custs, _ = risk_service.filter_customers(min_tenure=10, max_tenure=20, limit=10)
    for c in tenure_custs:
        assert 10 <= c["tenure"] <= 20

    # Test sorting descending by churn probability
    sorted_custs, _ = risk_service.filter_customers(limit=10, sort_by="churn_probability", ascending=False)
    probs = [c["churn_probability"] for c in sorted_custs]
    assert probs == sorted(probs, reverse=True)


def test_csv_export_format(risk_service):
    csv_str = risk_service.export_csv(risk_category="High", limit=20)
    assert "customerID" in csv_str
    assert "churn_probability" in csv_str
    assert "risk_category" in csv_str

    df_exported = pd.read_csv(io.StringIO(csv_str))
    assert len(df_exported) <= 20
    assert (df_exported["risk_category"] == "High").all()


def test_api_risk_distribution_endpoint():
    client = TestClient(api.app)
    response = client.get("/api/risk-distribution")
    assert response.status_code == 200
    data = response.json()
    assert "high" in data
    assert "medium" in data
    assert "low" in data
    assert data["high"]["count"] >= 0


def test_api_action_center_endpoints():
    client = TestClient(api.app)

    # Summary
    summary_resp = client.get("/api/action-center/summary")
    assert summary_resp.status_code == 200
    assert "high_risk_customers" in summary_resp.json()

    # Customers list with filters
    cust_resp = client.get("/api/action-center/customers", params={"risk_category": "High", "limit": 5})
    assert cust_resp.status_code == 200
    res = cust_resp.json()
    assert "customers" in res
    assert "total" in res
    assert len(res["customers"]) <= 5

    # CSV Export
    csv_resp = client.get("/api/action-center/export-csv", params={"risk_category": "High", "limit": 10})
    assert csv_resp.status_code == 200
    assert csv_resp.headers["content-type"].startswith("text/csv")
    assert "customerID" in csv_resp.text
