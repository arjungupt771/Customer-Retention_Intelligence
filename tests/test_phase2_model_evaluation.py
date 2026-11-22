"""Tests for Phase 2: Comprehensive Model Evaluation & Diagnostics."""
import numpy as np
import pytest
from fastapi.testclient import TestClient

import api
from churn_project.python.evaluation_service import (
    compute_classification_report,
    compute_comprehensive_diagnostics,
    compute_confusion_matrix,
    compute_lift_and_gain,
    compute_pr_curve,
    compute_roc_curve,
)


@pytest.fixture
def synthetic_eval_data():
    np.random.seed(42)
    # 100 samples with 30 churners
    y_true = np.array([1] * 30 + [0] * 70)
    # Probabilities higher for churners
    y_prob = np.concatenate([
        np.random.uniform(0.4, 0.95, size=30),
        np.random.uniform(0.05, 0.6, size=70),
    ])
    return y_true, y_prob


def test_confusion_matrix_calculation(synthetic_eval_data):
    y_true, y_prob = synthetic_eval_data
    cm_50 = compute_confusion_matrix(y_true, y_prob, threshold=0.50)
    assert "true_positives" in cm_50
    assert "true_negatives" in cm_50
    assert "false_positives" in cm_50
    assert "false_negatives" in cm_50
    assert cm_50["total"] == 100
    assert (
        cm_50["true_positives"]
        + cm_50["true_negatives"]
        + cm_50["false_positives"]
        + cm_50["false_negatives"]
        == 100
    )

    # Higher threshold should reduce or maintain true positives
    cm_80 = compute_confusion_matrix(y_true, y_prob, threshold=0.80)
    assert cm_80["true_positives"] <= cm_50["true_positives"]
    assert cm_80["false_positives"] <= cm_50["false_positives"]


def test_classification_report_metrics(synthetic_eval_data):
    y_true, y_prob = synthetic_eval_data
    report = compute_classification_report(y_true, y_prob, threshold=0.5)

    for metric in ("accuracy", "precision", "recall", "f1_score", "specificity", "balanced_accuracy"):
        assert metric in report
        assert 0.0 <= report[metric] <= 1.0

    # Verify specificity + FPR = 1
    cm = compute_confusion_matrix(y_true, y_prob, threshold=0.5)
    calc_spec = cm["true_negatives"] / (cm["true_negatives"] + cm["false_positives"])
    assert pytest.approx(report["specificity"], abs=1e-3) == calc_spec


def test_roc_curve_calculation(synthetic_eval_data):
    y_true, y_prob = synthetic_eval_data
    roc = compute_roc_curve(y_true, y_prob)

    assert "roc_auc" in roc
    assert 0.5 <= roc["roc_auc"] <= 1.0
    assert "points" in roc
    assert len(roc["points"]) > 0

    first_pt = roc["points"][0]
    assert "fpr" in first_pt
    assert "tpr" in first_pt
    assert "threshold" in first_pt


def test_precision_recall_curve_and_operating_point(synthetic_eval_data):
    y_true, y_prob = synthetic_eval_data
    pr = compute_pr_curve(y_true, y_prob, threshold=0.45)

    assert "average_precision" in pr
    assert 0.0 <= pr["average_precision"] <= 1.0
    assert "operating_metrics" in pr
    assert "precision" in pr["operating_metrics"]
    assert "recall" in pr["operating_metrics"]
    assert pr["selected_threshold"] == 0.45


def test_lift_and_gain_chart(synthetic_eval_data):
    y_true, y_prob = synthetic_eval_data
    lift_data = compute_lift_and_gain(y_true, y_prob, n_bins=10)

    assert "deciles" in lift_data
    assert len(lift_data["deciles"]) == 10
    assert lift_data["overall_churn_rate"] == 30.0

    # Top decile should have lift > 1.0 given our synthetic distribution
    top_decile = lift_data["deciles"][0]
    assert top_decile["decile"] == 1
    assert top_decile["lift"] >= 1.0

    # Last decile cumulative gain must be 100%
    last_decile = lift_data["deciles"][-1]
    assert last_decile["cumulative_churners_pct"] == 100.0
    assert last_decile["cumulative_customers_pct"] == 100.0


def test_comprehensive_diagnostics_service(synthetic_eval_data):
    y_true, y_prob = synthetic_eval_data
    diag = compute_comprehensive_diagnostics(y_true, y_prob, threshold=0.35)

    assert "confusion_matrix" in diag
    assert "classification_report" in diag
    assert "roc_curve" in diag
    assert "pr_curve" in diag
    assert "lift_chart" in diag
    assert diag["threshold"] == 0.35


def test_api_model_diagnostics_endpoint():
    client = TestClient(api.app)
    response = client.get("/api/model-diagnostics")
    assert response.status_code == 200
    data = response.json()

    assert "confusion_matrix" in data
    assert "classification_report" in data
    assert "roc_curve" in data
    assert "pr_curve" in data
    assert "lift_chart" in data
    assert data["confusion_matrix"]["total"] > 0

    # Test threshold parameter
    custom_resp = client.get("/api/model-diagnostics", params={"threshold": 0.20})
    assert custom_resp.status_code == 200
    assert custom_resp.json()["threshold"] == 0.20
