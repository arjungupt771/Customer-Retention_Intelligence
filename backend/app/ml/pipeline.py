"""Unified, reproducible machine learning training and evaluation pipeline."""
from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from backend.app.config import GLOBAL_CONFIG, PipelineConfig
from backend.app.ml.data.preprocessing import make_pipeline
from backend.app.ml.data.validation import (
    CATEGORICAL,
    NUMERIC,
    TARGET,
    clean_dataset,
    find_dataset,
)
from backend.app.ml.models.evaluate import evaluate_all_candidates
from backend.app.ml.models.persistence import save_model_artifact

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s]: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("ml_pipeline")


class ChurnTrainingPipeline:
    """Orchestrates end-to-end training, validation, evaluation, optimization, and persistence."""

    def __init__(self, config: PipelineConfig = GLOBAL_CONFIG) -> None:
        self.config = config
        self.start_time: float = 0.0

    def run(self, dataset_path: Path | str | None = None, artifact_dir: Path | str | None = None) -> dict[str, Any]:
        """Execute the complete training pipeline."""
        self.start_time = time.time()
        logger.info("=== Starting Churn Intelligence Training Pipeline ===")

        # Stage 1: Data Loading
        logger.info("[1/9] Loading dataset...")
        raw_path = Path(dataset_path) if dataset_path else find_dataset()
        raw_df = pd.read_csv(raw_path)
        logger.info("Loaded %d raw rows from %s", len(raw_df), raw_path)

        # Stage 2: Data Validation
        logger.info("[2/9] Validating dataset schema and integrity...")
        clean_df = clean_dataset(raw_df)
        dataset_info = {
            "path": str(raw_path),
            "rows": int(len(clean_df)),
            "columns": int(clean_df.shape[1]),
            "churn_rate": round(float(clean_df[TARGET].eq("Yes").mean() * 100), 2),
        }
        logger.info("Validation passed: %d records, churn rate: %.2f%%", dataset_info["rows"], dataset_info["churn_rate"])

        # Stage 3-6: Preprocessing, Candidate Training, Cross-Validation, Selection & Test Evaluation
        logger.info("[3-7/9] Running candidate cross-validation and test set evaluation...")
        evaluation_results = evaluate_all_candidates(clean_df, config=self.config)
        selected_model_name = evaluation_results["selected_model"]
        selected_metrics = evaluation_results["models"][selected_model_name]
        logger.info("Model selection completed. Winner: %s (CV PR-AUC: %.3f)",
                    selected_model_name, selected_metrics["cv"]["pr_auc"]["mean"])

        # Stage 8: Threshold Optimization
        logger.info("[8/9] Selecting optimal operating threshold...")
        optimal_threshold = float(selected_metrics["threshold"])
        logger.info("Optimal threshold chosen: %.3f", optimal_threshold)

        # Stage 9: Final Fitting and Persistence
        logger.info("[9/9] Fitting final model pipeline and persisting artifacts...")
        estimators = {
            "Logistic Regression": LogisticRegression(**self.config.models.logistic_regression),
            "Random Forest": RandomForestClassifier(**self.config.models.random_forest),
            "Gradient Boosting": GradientBoostingClassifier(**self.config.models.gradient_boosting),
        }
        final_estimator = estimators.get(
            selected_model_name,
            GradientBoostingClassifier(**self.config.models.gradient_boosting),
        )
        final_pipeline = make_pipeline(final_estimator)

        X = clean_df[CATEGORICAL + NUMERIC]
        y = clean_df[TARGET].eq("Yes").astype(int)
        final_pipeline.fit(X, y)

        elapsed = round(time.time() - self.start_time, 2)
        saved_dir = save_model_artifact(
            pipeline=final_pipeline,
            model_name=selected_model_name,
            metrics=evaluation_results,
            threshold=optimal_threshold,
            artifact_dir=artifact_dir,
            extra_metadata={
                "dataset_info": dataset_info,
                "pipeline_execution_time_seconds": elapsed,
            },
        )

        logger.info("=== Pipeline completed successfully in %.2f seconds! Artifacts saved to %s ===", elapsed, saved_dir)
        return {
            "status": "success",
            "selected_model": selected_model_name,
            "threshold": optimal_threshold,
            "metrics": evaluation_results,
            "execution_time_seconds": elapsed,
            "artifact_dir": str(saved_dir),
        }


def run_pipeline() -> None:
    pipeline = ChurnTrainingPipeline()
    pipeline.run()


if __name__ == "__main__":
    run_pipeline()
