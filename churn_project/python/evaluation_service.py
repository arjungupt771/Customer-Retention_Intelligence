"""Centralized comprehensive model evaluation and diagnostics service."""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

logger = logging.getLogger(__name__)


def compute_confusion_matrix(
    y_true: np.ndarray | list[int],
    y_prob: np.ndarray | list[float],
    threshold: float = 0.5,
) -> dict[str, int]:
    """Calculate TP, TN, FP, FN at configurable threshold."""
    y_t = np.asarray(y_true, dtype=int)
    y_p = np.asarray(y_prob, dtype=float)
    y_pred = (y_p >= threshold).astype(int)

    # Scikit-learn confusion_matrix returns [[TN, FP], [FN, TP]]
    tn, fp, fn, tp = confusion_matrix(y_t, y_pred, labels=[0, 1]).ravel()
    return {
        "true_positives": int(tp),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "total": int(len(y_t)),
    }


def compute_classification_report(
    y_true: np.ndarray | list[int],
    y_prob: np.ndarray | list[float],
    threshold: float = 0.5,
) -> dict[str, float]:
    """Calculate accuracy, precision, recall, f1, specificity, and balanced accuracy."""
    y_t = np.asarray(y_true, dtype=int)
    y_p = np.asarray(y_prob, dtype=float)
    y_pred = (y_p >= threshold).astype(int)

    cm = compute_confusion_matrix(y_t, y_p, threshold)
    tp, tn, fp, fn = cm["true_positives"], cm["true_negatives"], cm["false_positives"], cm["false_negatives"]

    precision = float(precision_score(y_t, y_pred, zero_division=0))
    recall = float(recall_score(y_t, y_pred, zero_division=0))
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    f1 = float(f1_score(y_t, y_pred, zero_division=0))
    accuracy = float(accuracy_score(y_t, y_pred))
    balanced_acc = float(balanced_accuracy_score(y_t, y_pred))

    return {
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "specificity": round(specificity, 4),
        "balanced_accuracy": round(balanced_acc, 4),
        "threshold": round(float(threshold), 3),
    }


def compute_roc_curve(
    y_true: np.ndarray | list[int],
    y_prob: np.ndarray | list[float],
    max_points: int = 50,
) -> dict[str, Any]:
    """Calculate ROC curve and ROC-AUC score, downsampled for visualization."""
    y_t = np.asarray(y_true, dtype=int)
    y_p = np.asarray(y_prob, dtype=float)

    auc_score = float(roc_auc_score(y_t, y_p))
    fpr, tpr, thresholds = roc_curve(y_t, y_p)

    # Downsample points for frontend charting if necessary
    total_pts = len(fpr)
    if total_pts > max_points:
        indices = np.linspace(0, total_pts - 1, max_points, dtype=int)
        fpr = fpr[indices]
        tpr = tpr[indices]
        thresholds = thresholds[indices]

    curve_points = []
    for f, t, th in zip(fpr, tpr, thresholds):
        th_val = 1.0 if not np.isfinite(th) or th > 1.0 else float(th)
        curve_points.append({
            "fpr": round(float(f), 4),
            "tpr": round(float(t), 4),
            "threshold": round(th_val, 4),
        })

    return {
        "roc_auc": round(auc_score, 4),
        "points": curve_points,
    }


def compute_pr_curve(
    y_true: np.ndarray | list[int],
    y_prob: np.ndarray | list[float],
    threshold: float = 0.5,
    max_points: int = 50,
) -> dict[str, Any]:
    """Generate PR curve, Average Precision, and metrics at selected threshold."""
    y_t = np.asarray(y_true, dtype=int)
    y_p = np.asarray(y_prob, dtype=float)

    avg_precision = float(average_precision_score(y_t, y_p))
    precisions, recalls, thresholds = precision_recall_curve(y_t, y_p)

    total_pts = len(thresholds)
    if total_pts > max_points:
        indices = np.linspace(0, total_pts - 1, max_points, dtype=int)
        p_sub = precisions[indices]
        r_sub = recalls[indices]
        th_sub = thresholds[indices]
    else:
        p_sub = precisions[:-1]
        r_sub = recalls[:-1]
        th_sub = thresholds

    curve_points = [
        {"precision": round(float(p), 4), "recall": round(float(r), 4), "threshold": round(float(th), 4)}
        for p, r, th in zip(p_sub, r_sub, th_sub)
    ]

    # Specific metrics at operating threshold
    y_pred = (y_p >= threshold).astype(int)
    operating_precision = float(precision_score(y_t, y_pred, zero_division=0))
    operating_recall = float(recall_score(y_t, y_pred, zero_division=0))

    return {
        "average_precision": round(avg_precision, 4),
        "selected_threshold": round(float(threshold), 3),
        "operating_metrics": {
            "precision": round(operating_precision, 4),
            "recall": round(operating_recall, 4),
        },
        "points": curve_points,
    }


def compute_lift_and_gain(
    y_true: np.ndarray | list[int],
    y_prob: np.ndarray | list[float],
    n_bins: int = 10,
) -> dict[str, Any]:
    """Calculate decile lift chart and cumulative gains."""
    y_t = np.asarray(y_true, dtype=int)
    y_p = np.asarray(y_prob, dtype=float)

    total_population = len(y_t)
    total_churners = int(np.sum(y_t))
    if total_population == 0 or total_churners == 0:
        return {"deciles": [], "overall_churn_rate": 0.0}

    overall_churn_rate = total_churners / total_population

    # Sort descending by probability
    sorted_idx = np.argsort(-y_p)
    sorted_y = y_t[sorted_idx]
    sorted_prob = y_p[sorted_idx]

    bin_size = total_population / n_bins
    deciles = []
    cum_churners = 0
    cum_customers = 0

    for i in range(n_bins):
        start_idx = int(round(i * bin_size))
        end_idx = int(round((i + 1) * bin_size))
        bin_y = sorted_y[start_idx:end_idx]
        bin_p = sorted_prob[start_idx:end_idx]

        count = len(bin_y)
        churners = int(np.sum(bin_y))
        cum_churners += churners
        cum_customers += count

        decile_churn_rate = churners / count if count > 0 else 0.0
        lift = decile_churn_rate / overall_churn_rate if overall_churn_rate > 0 else 0.0
        cum_gain_pct = (cum_churners / total_churners * 100.0) if total_churners > 0 else 0.0
        cum_pop_pct = (cum_customers / total_population * 100.0)

        deciles.append({
            "decile": i + 1,
            "customers": count,
            "churners": churners,
            "churn_rate": round(decile_churn_rate * 100, 2),
            "min_probability": round(float(np.min(bin_p)), 4) if count else 0.0,
            "max_probability": round(float(np.max(bin_p)), 4) if count else 0.0,
            "lift": round(lift, 2),
            "cumulative_customers_pct": round(cum_pop_pct, 1),
            "cumulative_churners_pct": round(cum_gain_pct, 1),
            "gain_pct": round(cum_gain_pct, 1),
        })

    return {
        "overall_churn_rate": round(overall_churn_rate * 100, 2),
        "total_population": total_population,
        "total_churners": total_churners,
        "deciles": deciles,
    }


def compute_comprehensive_diagnostics(
    y_true: np.ndarray | list[int],
    y_prob: np.ndarray | list[float],
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Central entrypoint returning all evaluation diagnostics."""
    return {
        "threshold": round(float(threshold), 3),
        "confusion_matrix": compute_confusion_matrix(y_true, y_prob, threshold),
        "classification_report": compute_classification_report(y_true, y_prob, threshold),
        "roc_curve": compute_roc_curve(y_true, y_prob),
        "pr_curve": compute_pr_curve(y_true, y_prob, threshold),
        "lift_chart": compute_lift_and_gain(y_true, y_prob),
    }
