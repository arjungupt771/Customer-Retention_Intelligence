"""Statistical confidence intervals and model comparison via bootstrap resampling."""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)

DEFAULT_BOOTSTRAP_CONFIG = {
    "n_bootstraps": 500,
    "confidence_level": 0.95,
    "random_seed": 42,
}


def compute_bootstrap_confidence_intervals(
    y_true: np.ndarray | list[int],
    y_prob: np.ndarray | list[float],
    threshold: float = 0.5,
    n_bootstraps: int = 500,
    confidence_level: float = 0.95,
    random_seed: int = 42,
) -> dict[str, Any]:
    """Calculate point estimates and percentile bootstrap confidence intervals."""
    y_t = np.asarray(y_true, dtype=int)
    y_p = np.asarray(y_prob, dtype=float)
    n_samples = len(y_t)

    if n_samples < 5:
        raise ValueError("Bootstrap requires at least 5 samples")

    rng = np.random.default_rng(random_seed)

    # Point estimates on full sample
    y_pred_full = (y_p >= threshold).astype(int)
    point_estimates = {
        "roc_auc": float(roc_auc_score(y_t, y_p)) if len(np.unique(y_t)) > 1 else 0.5,
        "pr_auc": float(average_precision_score(y_t, y_p)),
        "precision": float(precision_score(y_t, y_pred_full, zero_division=0)),
        "recall": float(recall_score(y_t, y_pred_full, zero_division=0)),
        "f1_score": float(f1_score(y_t, y_pred_full, zero_division=0)),
        "accuracy": float(accuracy_score(y_t, y_pred_full)),
    }

    boot_metrics: dict[str, list[float]] = {k: [] for k in point_estimates}

    for _ in range(n_bootstraps):
        idx = rng.choice(n_samples, size=n_samples, replace=True)
        sample_yt = y_t[idx]
        sample_yp = y_p[idx]

        # Ensure both classes are present for ROC-AUC
        if len(np.unique(sample_yt)) < 2:
            continue

        sample_pred = (sample_yp >= threshold).astype(int)

        try:
            boot_metrics["roc_auc"].append(float(roc_auc_score(sample_yt, sample_yp)))
            boot_metrics["pr_auc"].append(float(average_precision_score(sample_yt, sample_yp)))
            boot_metrics["precision"].append(float(precision_score(sample_yt, sample_pred, zero_division=0)))
            boot_metrics["recall"].append(float(recall_score(sample_yt, sample_pred, zero_division=0)))
            boot_metrics["f1_score"].append(float(f1_score(sample_yt, sample_pred, zero_division=0)))
            boot_metrics["accuracy"].append(float(accuracy_score(sample_yt, sample_pred)))
        except Exception:
            continue

    alpha = float(confidence_level)
    lower_pct = (1.0 - alpha) / 2.0 * 100.0
    upper_pct = (1.0 + alpha) / 2.0 * 100.0

    result: dict[str, Any] = {
        "confidence_level": alpha,
        "n_bootstraps": len(boot_metrics["roc_auc"]),
        "threshold": round(float(threshold), 3),
        "metrics": {},
    }

    for metric_name, point_est in point_estimates.items():
        vals = boot_metrics[metric_name]
        if vals:
            lower = float(np.percentile(vals, lower_pct))
            upper = float(np.percentile(vals, upper_pct))
        else:
            lower = point_est
            upper = point_est

        result["metrics"][metric_name] = {
            "estimate": round(point_est, 3),
            "confidence_interval": {
                "lower": round(lower, 3),
                "upper": round(upper, 3),
            },
        }

    return result


def compare_models_bootstrap(
    y_true: np.ndarray | list[int],
    prob_a: np.ndarray | list[float],
    prob_b: np.ndarray | list[float],
    model_a_name: str = "Model A",
    model_b_name: str = "Model B",
    threshold_a: float = 0.5,
    threshold_b: float = 0.5,
    n_bootstraps: int = 500,
    confidence_level: float = 0.95,
    random_seed: int = 42,
) -> dict[str, Any]:
    """Compare two models using paired bootstrap resampling."""
    y_t = np.asarray(y_true, dtype=int)
    p_a = np.asarray(prob_a, dtype=float)
    p_b = np.asarray(prob_b, dtype=float)
    n_samples = len(y_t)

    rng = np.random.default_rng(random_seed)

    diffs = {"roc_auc": [], "pr_auc": [], "f1_score": []}

    for _ in range(n_bootstraps):
        idx = rng.choice(n_samples, size=n_samples, replace=True)
        s_yt = y_t[idx]
        if len(np.unique(s_yt)) < 2:
            continue

        s_pa, s_pb = p_a[idx], p_b[idx]
        pred_a = (s_pa >= threshold_a).astype(int)
        pred_b = (s_pb >= threshold_b).astype(int)

        diffs["roc_auc"].append(float(roc_auc_score(s_yt, s_pa) - roc_auc_score(s_yt, s_pb)))
        diffs["pr_auc"].append(float(average_precision_score(s_yt, s_pa) - average_precision_score(s_yt, s_pb)))
        diffs["f1_score"].append(float(f1_score(s_yt, pred_a, zero_division=0) - f1_score(s_yt, pred_b, zero_division=0)))

    alpha = float(confidence_level)
    lower_pct = (1.0 - alpha) / 2.0 * 100.0
    upper_pct = (1.0 + alpha) / 2.0 * 100.0

    comparisons: dict[str, Any] = {}
    for metric_name, d_vals in diffs.items():
        if d_vals:
            point_diff = float(np.mean(d_vals))
            ci_lower = float(np.percentile(d_vals, lower_pct))
            ci_upper = float(np.percentile(d_vals, upper_pct))
        else:
            point_diff = 0.0
            ci_lower = 0.0
            ci_upper = 0.0

        # Check statistical interpretation without overstating claims
        includes_zero = bool(ci_lower <= 0 <= ci_upper)
        if includes_zero:
            interpretation = (
                f"No statistically significant difference in {metric_name} "
                f"at the {int(alpha*100)}% confidence level (CI spans zero)."
            )
        else:
            leader = model_a_name if point_diff > 0 else model_b_name
            interpretation = (
                f"{leader} has a statistically significant advantage in {metric_name} "
                f"(difference: {point_diff:+.3f}, {int(alpha*100)}% CI: [{ci_lower:.3f}, {ci_upper:.3f}])."
            )

        comparisons[metric_name] = {
            "difference_estimate": round(point_diff, 3),
            "confidence_interval": {"lower": round(ci_lower, 3), "upper": round(ci_upper, 3)},
            "statistically_significant": not includes_zero,
            "interpretation": interpretation,
        }

    return {
        "model_a": model_a_name,
        "model_b": model_b_name,
        "confidence_level": alpha,
        "comparisons": comparisons,
    }
