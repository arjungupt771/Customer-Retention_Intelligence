import sqlite3
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from churn_project.python.churn_analysis import (build_model_evaluation,
    clean_dataset, key_metrics, make_pipeline)
import api

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "churn_project" / "data" / "telco_churn.csv"


@pytest.fixture(scope="module")
def dataset():
    return clean_dataset(pd.read_csv(DATA))


def test_validation_rejects_duplicate_customer_ids(dataset):
    duplicated = pd.concat([dataset.iloc[[0]], dataset.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate customerID"):
        clean_dataset(duplicated)


def test_validation_rejects_invalid_target(dataset):
    invalid = dataset.iloc[[0]].copy()
    invalid.loc[:, "Churn"] = "Maybe"
    with pytest.raises(ValueError, match="Unexpected Churn"):
        clean_dataset(invalid)


def test_validation_rejects_empty_dataset(dataset):
    with pytest.raises(ValueError, match="at least one customer"):
        clean_dataset(dataset.iloc[:0])


def test_validation_rejects_unexpected_category(dataset):
    invalid = dataset.iloc[[0]].copy()
    invalid.loc[:, "Contract"] = "Lifetime"
    with pytest.raises(ValueError, match="Unexpected Contract"):
        clean_dataset(invalid)


def test_business_metrics_match_known_small_fixture(dataset):
    sample = dataset.iloc[[0, 1]].copy()
    sample.loc[:, "Churn"] = ["Yes", "No"]
    sample.loc[:, "MonthlyCharges"] = [30.0, 70.0]
    metrics = key_metrics(sample)
    assert metrics["total_customers"] == 2
    assert metrics["churned_customers"] == 1
    assert metrics["active_customers"] == 1
    assert metrics["churn_rate"] == 50.0
    assert metrics["historical_churned_monthly_revenue"] == 30.0
    assert metrics["active_monthly_revenue_at_risk"] == 70.0


def test_sql_kpi_query_matches_python(dataset):
    sql = (ROOT / "churn_project" / "sql" / "churn_analysis.sql").read_text(encoding="utf-8")
    first_query = "SELECT" + sql.split("\nSELECT", 1)[1].split(";", 1)[0]
    with sqlite3.connect(":memory:") as connection:
        dataset.to_sql("telco_churn", connection, index=False)
        sql_metrics = pd.read_sql_query(first_query, connection).iloc[0]
        connection.executescript(sql)
    python_metrics = key_metrics(dataset)
    assert int(sql_metrics.total_customers) == python_metrics["total_customers"]
    assert int(sql_metrics.churned_customers) == python_metrics["churned_customers"]
    assert int(sql_metrics.active_customers) == python_metrics["active_customers"]
    assert sql_metrics.churn_rate_pct == pytest.approx(python_metrics["churn_rate"], abs=0.01)
    assert sql_metrics.monthly_revenue_lost == pytest.approx(python_metrics["historical_churned_monthly_revenue"])


def test_pipeline_one_hot_encodes_nominal_categories(dataset):
    pipeline = make_pipeline(__import__("sklearn.linear_model", fromlist=["LogisticRegression"]).LogisticRegression())
    small = dataset.iloc[:100]
    pipeline.fit(small[["gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
                        "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
                        "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
                        "PaperlessBilling", "PaymentMethod", "SeniorCitizen", "tenure",
                        "MonthlyCharges", "TotalCharges"]], small.Churn.eq("Yes").astype(int))
    assert pipeline.predict_proba(small[["gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
                        "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
                        "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
                        "PaperlessBilling", "PaymentMethod", "SeniorCitizen", "tenure",
                        "MonthlyCharges", "TotalCharges"]]).shape == (100, 2)


def test_api_health_metrics_and_filter_validation():
    client = TestClient(api.app)
    assert client.get("/health").json() == {"status": "ok"}
    response = client.get("/api/key-metrics")
    assert response.status_code == 200
    assert response.json()["total_customers"] > 0
    filtered = client.get("/api/segments", params={"contract": "Month-to-month", "tenure_group": "0–12 months"})
    assert filtered.status_code == 200
    assert filtered.json()["count"] > 0
    assert 0 <= filtered.json()["metrics"]["churn_rate"] <= 100
    bad = client.get("/api/segments", params={"churn": "sometimes"})
    assert bad.status_code == 422


def test_api_missing_file_is_a_readable_readiness_error(monkeypatch):
    client = TestClient(api.app)
    monkeypatch.setattr(api, "find_dataset", lambda _root: (_ for _ in ()).throw(FileNotFoundError("dataset missing")))
    api.load_dataset.cache_clear()
    try:
        response = client.get("/ready")
        assert response.status_code == 503
        assert response.json()["detail"] == "dataset missing"
    finally:
        api.load_dataset.cache_clear()


def test_model_evaluation_rejects_single_class(dataset):
    single_class = dataset.iloc[:50].copy()
    single_class.loc[:, "Churn"] = "No"
    with pytest.raises(ValueError, match="two rows in each target class"):
        build_model_evaluation(single_class)


def test_model_evaluation_is_reproducible(dataset):
    sample = dataset.groupby("Churn", group_keys=False).sample(n=120, random_state=17).reset_index(drop=True)
    first = build_model_evaluation(sample)
    second = build_model_evaluation(sample)
    assert first["selected_model"] == second["selected_model"]
    assert first["baseline"] == second["baseline"]
    assert first["models"] == second["models"]
