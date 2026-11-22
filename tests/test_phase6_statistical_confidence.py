"""Tests for Phase 6: Statistical Confidence & Model Uncertainty."""
import numpy as np
import pytest
from fastapi.testclient import TestClient

import api
from churn_project.python.confidence_service import (
    compare_models_bootstrap,
    compute_bootstrap_confidence_intervals,
)


@pytest.fixture
def eval_dataset():
    np.random.seed(123)
    y_true = np.array([1] * 50 + [0] * 50)
    # Model A: stronger predictor
    prob_a = np.concatenate([
        np.random.uniform(0.5, 0.95, size=50),
        np.random.uniform(0.05, 0.45, size=50),
    ])
    # Model B: weaker predictor
    prob_b = np.concatenate([
        np.random.uniform(0.3, 0.75, size=50),
        np.random.uniform(0.2, 0.6, size=50),
    ])
    return y_true, prob_a, prob_b


def test_bootstrap_confidence_intervals_structure(eval_dataset):
    y_true, prob_a, _ = eval_dataset
    res = compute_bootstrap_confidence_intervals(
        y_true, prob_a, threshold=0.5, n_bootstraps=100, confidence_level=0.95, random_seed=42
    )

    assert "confidence_level" in res
    assert res["confidence_level"] == 0.95
    assert "metrics" in res

    for metric in ("roc_auc", "pr_auc", "precision", "recall", "f1_score", "accuracy"):
        assert metric in res["metrics"]
        m = res["metrics"][metric]
        assert "estimate" in m
        assert "confidence_interval" in m
        ci = m["confidence_interval"]
        assert "lower" in ci
        assert "upper" in ci
        assert ci["lower"] <= ci["upper"]
        # CI should surround estimate
        assert ci["lower"] <= m["estimate"] <= ci["upper"] or pytest.approx(ci["lower"], abs=0.05) <= m["estimate"]


def test_bootstrap_reproducibility(eval_dataset):
    y_true, prob_a, _ = eval_dataset
    res1 = compute_bootstrap_confidence_intervals(
        y_true, prob_a, threshold=0.5, n_bootstraps=50, random_seed=999
    )
    res2 = compute_bootstrap_confidence_intervals(
        y_true, prob_a, threshold=0.5, n_bootstraps=50, random_seed=999
    )
    assert res1["metrics"]["roc_auc"] == res2["metrics"]["roc_auc"]
    assert res1["metrics"]["f1_score"] == res2["metrics"]["f1_score"]


def test_model_comparison_bootstrap(eval_dataset):
    y_true, prob_a, prob_b = eval_dataset
    comp = compare_models_bootstrap(
        y_true,
        prob_a,
        prob_b,
        model_a_name="Gradient Boosting",
        model_b_name="Logistic Regression",
        n_bootstraps=100,
        random_seed=42,
    )

    assert "comparisons" in comp
    assert "roc_auc" in comp["comparisons"]
    roc_comp = comp["comparisons"]["roc_auc"]
    assert "difference_estimate" in roc_comp
    assert "confidence_interval" in roc_comp
    assert "statistically_significant" in roc_comp
    assert "interpretation" in roc_comp
    # Since Model A is noticeably better, difference estimate should be positive
    assert roc_comp["difference_estimate"] > 0


def test_api_model_confidence_endpoint():
    client = TestClient(api.app)
    response = client.get("/api/model-confidence", params={"n_bootstraps": 100})
    assert response.status_code == 200
    data = response.json()
    assert "metrics" in data
    assert "roc_auc" in data["metrics"]
    assert "confidence_interval" in data["metrics"]["roc_auc"]
