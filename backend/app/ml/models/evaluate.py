"""Model cross-validation and candidate evaluation module."""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split

from backend.app.config import GLOBAL_CONFIG
from backend.app.ml.data.preprocessing import make_pipeline
from backend.app.ml.data.validation import CATEGORICAL, NUMERIC, TARGET
from backend.app.ml.models.train import get_model_candidates

logger = logging.getLogger(__name__)


def evaluate_all_candidates(df: pd.DataFrame, config=GLOBAL_CONFIG) -> dict[str, Any]:
    """Execute leakage-safe cross-validation and candidate selection."""
    X = df[CATEGORICAL + NUMERIC]
    y = df[TARGET].eq("Yes").astype(int)

    if y.nunique() != 2 or y.value_counts().min() < 2:
        raise ValueError("Model evaluation requires at least two rows in each target class")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config.training.test_size,
        random_state=config.training.random_seed,
        stratify=y,
    )

    baseline_prob = float(y_train.mean())
    baseline_pred = np.full(len(y_test), int(baseline_prob >= 0.5))

    candidates = get_model_candidates(config)
    n_splits = min(config.training.cv_splits, int(y_train.value_counts().min()))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=config.training.random_seed)

    scoring_metrics = {
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
        "pr_auc": "average_precision",
    }

    evaluated = {}
    fitted = {}

    for name, estimator in candidates.items():
        pipeline = make_pipeline(estimator)
        cv_result = cross_validate(pipeline, X_train, y_train, cv=cv, scoring=scoring_metrics, n_jobs=1)

        X_fit, X_val, y_fit, y_val = train_test_split(
            X_train, y_train,
            test_size=config.training.validation_split,
            random_state=314,
            stratify=y_train,
        )
        threshold_model = make_pipeline(estimator).fit(X_fit, y_fit)
        val_probs = threshold_model.predict_proba(X_val)[:, 1]

        # Cost-optimal threshold selection on validation split
        choices = np.linspace(0.05, 0.95, 91)
        threshold = float(min(
            choices,
            key=lambda t: 5 * np.sum((y_val.to_numpy() == 1) & (val_probs < t))
            + np.sum((y_val.to_numpy() == 0) & (val_probs >= t)),
        ))

        pipeline.fit(X_train, y_train)
        test_probs = pipeline.predict_proba(X_test)[:, 1]
        prediction = (test_probs >= threshold).astype(int)

        evaluated[name] = {
            "precision": round(float(precision_score(y_test, prediction, zero_division=0)), 3),
            "recall": round(float(recall_score(y_test, prediction, zero_division=0)), 3),
            "f1_score": round(float(f1_score(y_test, prediction, zero_division=0)), 3),
            "accuracy": round(float(accuracy_score(y_test, prediction)), 3),
            "auc_roc": round(float(roc_auc_score(y_test, test_probs)), 3),
            "pr_auc": round(float(average_precision_score(y_test, test_probs)), 3),
            "threshold": round(threshold, 3),
            "cv": {
                k: {
                    "mean": round(float(np.mean(cv_result[f"test_{k}"])), 3),
                    "std": round(float(np.std(cv_result[f"test_{k}"])), 3),
                }
                for k in scoring_metrics
            },
            "test_population": int(len(y_test)),
            "test_churners": int(y_test.sum()),
        }
        fitted[name] = pipeline

    selected = max(evaluated, key=lambda n: evaluated[n]["cv"]["pr_auc"]["mean"])
    model = fitted[selected]

    try:
        names = model.named_steps["preprocess"].get_feature_names_out()
        est = model.named_steps["model"]
        weights = getattr(est, "feature_importances_", None)
        if weights is None and hasattr(est, "coef_"):
            weights = np.abs(est.coef_[0])
        importance = [] if weights is None else [
            {"feature": str(n).replace("categorical__", "").replace("numeric__", ""),
             "importance": round(float(w), 5),
             "interpretation": "association, not causation"}
            for n, w in sorted(zip(names, weights), key=lambda pair: pair[1], reverse=True)[:10]
        ]
    except Exception:
        importance = []

    baseline_probabilities = np.full(len(y_test), baseline_prob)
    baseline = {
        "precision": round(float(precision_score(y_test, baseline_pred, zero_division=0)), 3),
        "recall": round(float(recall_score(y_test, baseline_pred, zero_division=0)), 3),
        "f1_score": round(float(f1_score(y_test, baseline_pred, zero_division=0)), 3),
        "accuracy": round(float(accuracy_score(y_test, baseline_pred)), 3),
        "auc_roc": round(float(roc_auc_score(y_test, baseline_probabilities)), 3),
        "pr_auc": round(float(average_precision_score(y_test, baseline_probabilities)), 3),
    }

    return {
        "baseline": baseline,
        "models": evaluated,
        "selected_model": selected,
        "feature_importance": importance,
        "evaluation_note": "Stratified held-out test set; threshold selected on training data only.",
        "threshold_cost_assumption": {
            "missed_churner_cost": config.costs.missed_churner_cost,
            "false_positive_outreach_cost": config.costs.false_positive_cost,
            "description": "Configurable cost parameters.",
        },
    }
