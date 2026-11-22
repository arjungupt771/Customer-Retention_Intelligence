"""Cost-sensitive classification threshold optimization and evaluation module."""
from __future__ import annotations

import json
import logging
from typing import Any

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score

from churn_project.python.persistence import CHURN_MODEL_DIR

logger = logging.getLogger(__name__)

DEFAULT_COST_CONFIG = {
    "missed_churner_cost": 500.0,  # Revenue/LTV lost per undetected churner
    "false_positive_cost": 50.0,   # Wasted outreach / operational overhead
    "intervention_cost": 25.0,     # Retention offer/discount cost per outreach
    "description": "Configurable illustrative cost assumptions for business optimization.",
}


class ThresholdOptimizer:
    """Evaluates and optimizes classification thresholds against business economics."""

    def __init__(
        self,
        y_true: np.ndarray | list[int],
        y_prob: np.ndarray | list[float],
        cost_config: dict[str, Any] | None = None,
    ) -> None:
        self.y_true = np.asarray(y_true, dtype=int)
        self.y_prob = np.asarray(y_prob, dtype=float)
        self.cost_config = dict(DEFAULT_COST_CONFIG)
        if cost_config:
            self.cost_config.update(cost_config)

    def evaluate_threshold(self, threshold: float) -> dict[str, Any]:
        """Calculate statistical and economic metrics at a specific operating threshold."""
        t = float(threshold)
        y_pred = (self.y_prob >= t).astype(int)

        tp = int(np.sum((self.y_true == 1) & (y_pred == 1)))
        fp = int(np.sum((self.y_true == 0) & (y_pred == 1)))
        fn = int(np.sum((self.y_true == 1) & (y_pred == 0)))
        tn = int(np.sum((self.y_true == 0) & (y_pred == 0)))

        total = len(self.y_true)
        flagged = tp + fp
        total_positives = tp + fn
        total_negatives = tn + fp

        precision = float(tp / flagged) if flagged > 0 else 0.0
        recall = float(tp / total_positives) if total_positives > 0 else 0.0
        f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        fpr = float(fp / total_negatives) if total_negatives > 0 else 0.0
        fnr = float(fn / total_positives) if total_positives > 0 else 0.0

        missed_cost_unit = float(self.cost_config["missed_churner_cost"])
        fp_cost_unit = float(self.cost_config["false_positive_cost"])
        interv_cost_unit = float(self.cost_config["intervention_cost"])

        expected_missed_cost = fn * missed_cost_unit
        expected_fp_cost = fp * fp_cost_unit
        expected_intervention_cost = flagged * interv_cost_unit
        total_cost = expected_missed_cost + expected_fp_cost + expected_intervention_cost

        return {
            "threshold": round(t, 3),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
            "false_positive_rate": round(fpr, 4),
            "false_negative_rate": round(fnr, 4),
            "customers_flagged": flagged,
            "flagged_percentage": round((flagged / total) * 100, 2) if total else 0.0,
            "churners_identified": tp,
            "true_negatives": tn,
            "false_positives": fp,
            "false_negatives": fn,
            "expected_missed_churn_cost": round(expected_missed_cost, 2),
            "expected_false_positive_cost": round(expected_fp_cost, 2),
            "expected_intervention_cost": round(expected_intervention_cost, 2),
            "total_estimated_cost": round(total_cost, 2),
        }

    def sweep_thresholds(
        self,
        min_threshold: float = 0.05,
        max_threshold: float = 0.95,
        step: float = 0.05,
    ) -> list[dict[str, Any]]:
        """Evaluate a range of thresholds across the operating spectrum."""
        thresholds = np.arange(min_threshold, max_threshold + 1e-5, step)
        return [self.evaluate_threshold(t) for t in thresholds]

    def optimize(
        self,
        min_threshold: float = 0.05,
        max_threshold: float = 0.95,
        step: float = 0.05,
        budget_max_flagged_pct: float = 30.0,
        target_recall: float = 0.70,
    ) -> dict[str, Any]:
        """Identify key optimal operating points for business decisions."""
        curve = self.sweep_thresholds(min_threshold, max_threshold, step)

        # 1. Minimum Total Cost
        min_cost_point = min(curve, key=lambda x: x["total_estimated_cost"])

        # 2. Maximum F1 Score
        max_f1_point = max(curve, key=lambda x: x["f1_score"])

        # 3. Maximum Recall under budget constraint (e.g. at most 30% customers flagged)
        budget_eligible = [pt for pt in curve if pt["flagged_percentage"] <= budget_max_flagged_pct]
        max_recall_budget_point = (
            max(budget_eligible, key=lambda x: x["recall"]) if budget_eligible else curve[-1]
        )

        # 4. Maximum Precision given a recall requirement (e.g. recall >= 70%)
        recall_eligible = [pt for pt in curve if pt["recall"] >= target_recall]
        max_precision_recall_point = (
            max(recall_eligible, key=lambda x: x["precision"]) if recall_eligible else curve[0]
        )

        return {
            "cost_config": self.cost_config,
            "optimal_thresholds": {
                "min_cost": min_cost_point,
                "max_f1": max_f1_point,
                "max_recall_under_budget": max_recall_budget_point,
                "max_precision_at_target_recall": max_precision_recall_point,
            },
            "threshold_curve": curve,
        }


def persist_selected_threshold(
    new_threshold: float,
    artifact_dir=None,
) -> dict[str, Any]:
    """Persist updated operating threshold to metadata.json."""
    from pathlib import Path
    target_dir = Path(artifact_dir) if artifact_dir else CHURN_MODEL_DIR
    meta_path = target_dir / "metadata.json"

    if not meta_path.is_file():
        raise FileNotFoundError(f"metadata.json not found at {meta_path}")

    with open(meta_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    metadata["threshold"] = round(float(new_threshold), 3)

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    logger.info("Persisted new operating threshold %.3f to %s", new_threshold, meta_path)
    return metadata
