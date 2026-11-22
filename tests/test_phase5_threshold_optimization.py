"""Tests for Phase 5: Cost-Sensitive Threshold Optimization."""
import numpy as np
import pytest
from fastapi.testclient import TestClient

import api
from churn_project.python.threshold_optimizer import (
    ThresholdOptimizer,
    persist_selected_threshold,
)


@pytest.fixture
def eval_distribution():
    np.random.seed(42)
    y_true = np.array([1] * 40 + [0] * 60)
    y_prob = np.concatenate([
        np.random.uniform(0.3, 0.9, size=40),
        np.random.uniform(0.05, 0.5, size=60),
    ])
    return y_true, y_prob


def test_threshold_cost_calculation(eval_distribution):
    y_true, y_prob = eval_distribution
    optimizer = ThresholdOptimizer(
        y_true,
        y_prob,
        cost_config={
            "missed_churner_cost": 200.0,
            "false_positive_cost": 40.0,
            "intervention_cost": 10.0,
        },
    )

    eval_pt = optimizer.evaluate_threshold(0.3)
    assert "threshold" in eval_pt
    assert "precision" in eval_pt
    assert "recall" in eval_pt
    assert "f1_score" in eval_pt
    assert "false_positive_rate" in eval_pt
    assert "false_negative_rate" in eval_pt
    assert "customers_flagged" in eval_pt
    assert "churners_identified" in eval_pt
    assert "expected_missed_churn_cost" in eval_pt
    assert "expected_false_positive_cost" in eval_pt
    assert "expected_intervention_cost" in eval_pt
    assert "total_estimated_cost" in eval_pt

    # Total cost must equal sum of components
    expected_total = (
        eval_pt["expected_missed_churn_cost"]
        + eval_pt["expected_false_positive_cost"]
        + eval_pt["expected_intervention_cost"]
    )
    assert pytest.approx(eval_pt["total_estimated_cost"], abs=0.1) == expected_total


def test_threshold_optimization_targets(eval_distribution):
    y_true, y_prob = eval_distribution
    optimizer = ThresholdOptimizer(y_true, y_prob)
    opt = optimizer.optimize(min_threshold=0.1, max_threshold=0.9, step=0.1)

    assert "cost_config" in opt
    assert "optimal_thresholds" in opt
    assert "min_cost" in opt["optimal_thresholds"]
    assert "max_f1" in opt["optimal_thresholds"]
    assert "max_recall_under_budget" in opt["optimal_thresholds"]
    assert "max_precision_at_target_recall" in opt["optimal_thresholds"]

    min_cost = opt["optimal_thresholds"]["min_cost"]
    curve = opt["threshold_curve"]
    assert all(min_cost["total_estimated_cost"] <= pt["total_estimated_cost"] for pt in curve)


def test_threshold_persistence(tmp_path):
    meta_file = tmp_path / "metadata.json"
    meta_file.write_text('{"threshold": 0.50, "model_name": "Test"}', encoding="utf-8")

    updated = persist_selected_threshold(0.28, artifact_dir=tmp_path)
    assert updated["threshold"] == 0.28

    with open(meta_file, encoding="utf-8") as f:
        data = __import__("json").load(f)
    assert data["threshold"] == 0.28


def test_api_threshold_analysis_endpoint():
    client = TestClient(api.app)
    response = client.get("/api/threshold-analysis", params={"step": 0.1})
    assert response.status_code == 200
    data = response.json()

    assert "cost_config" in data
    assert "current_threshold" in data
    assert "optimal_thresholds" in data
    assert "threshold_curve" in data
    assert len(data["threshold_curve"]) > 0


def test_api_set_operating_threshold_endpoint():
    client = TestClient(api.app)
    set_resp = client.post("/api/threshold/set", json={"threshold": 0.22})
    assert set_resp.status_code == 200
    assert set_resp.json()["threshold"] == 0.22

    # Verify that predict endpoint now uses updated threshold
    pred_resp = client.post("/api/predict", json={"tenure": 10, "monthly_charges": 60.0})
    assert pred_resp.status_code == 200
    assert pred_resp.json()["threshold"] == 0.22
