"""Model persistence and artifact management for churn intelligence."""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from churn_project.python.churn_analysis import (
    CATEGORICAL,
    NUMERIC,
    TARGET,
    build_model_evaluation,
    clean_dataset,
    find_dataset,
    make_pipeline,
)

logger = logging.getLogger(__name__)

DEFAULT_ARTIFACT_ROOT = Path(__file__).resolve().parents[2] / "artifacts" / "models"
CHURN_MODEL_DIR = DEFAULT_ARTIFACT_ROOT / "churn_model"


def save_model_artifact(
    pipeline: Pipeline,
    model_name: str,
    metrics: dict[str, Any],
    threshold: float,
    artifact_dir: Path | str | None = None,
    version: str = "1.0.0",
    extra_metadata: dict[str, Any] | None = None,
) -> Path:
    """Persist pipeline, metadata, and evaluation metrics in versioned directory."""
    target_dir = Path(artifact_dir) if artifact_dir else CHURN_MODEL_DIR
    target_dir.mkdir(parents=True, exist_ok=True)

    # 1. Save model pipeline with joblib
    model_path = target_dir / "model.joblib"
    joblib.dump(pipeline, model_path, compress=3)

    # 2. Metadata
    metadata = {
        "model_name": model_name,
        "version": version,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "threshold": float(threshold),
        "target": TARGET,
        "categorical_features": CATEGORICAL,
        "numeric_features": NUMERIC,
        "required_features": CATEGORICAL + NUMERIC,
        "risk_categories": {
            "high": {"min_probability": 0.70},
            "medium": {"min_probability": 0.30},
            "low": {"min_probability": 0.00},
        },
        "sklearn_version": joblib.__version__,
        **(extra_metadata or {}),
    }
    metadata_path = target_dir / "metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    # 3. Metrics
    metrics_path = target_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    logger.info("Saved model artifact to %s (model: %s, threshold: %.3f)", target_dir, model_name, threshold)
    return target_dir


def load_model_artifact(
    artifact_dir: Path | str | None = None,
) -> Tuple[Pipeline, dict[str, Any], dict[str, Any]]:
    """Load model pipeline, metadata, and metrics from artifact directory."""
    target_dir = Path(artifact_dir) if artifact_dir else CHURN_MODEL_DIR
    model_path = target_dir / "model.joblib"
    metadata_path = target_dir / "metadata.json"
    metrics_path = target_dir / "metrics.json"

    if not model_path.is_file():
        raise FileNotFoundError(f"Model artifact not found at {model_path}")
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Metadata not found at {metadata_path}")
    if not metrics_path.is_file():
        raise FileNotFoundError(f"Metrics not found at {metrics_path}")

    try:
        pipeline = joblib.load(model_path)
    except Exception as exc:
        raise RuntimeError(f"Corrupted or incompatible model artifact at {model_path}: {exc}") from exc

    try:
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
    except Exception as exc:
        raise RuntimeError(f"Corrupted metadata file at {metadata_path}: {exc}") from exc

    try:
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)
    except Exception as exc:
        raise RuntimeError(f"Corrupted metrics file at {metrics_path}: {exc}") from exc

    return pipeline, metadata, metrics


def train_and_persist_selected_model(
    dataset_path: Path | str | None = None,
    artifact_dir: Path | str | None = None,
    version: str = "1.0.0",
) -> Tuple[Pipeline, dict[str, Any], dict[str, Any]]:
    """Run model evaluation pipeline, fit selected candidate, and persist artifacts."""
    csv_path = Path(dataset_path) if dataset_path else find_dataset()
    df = clean_dataset(pd.read_csv(csv_path))

    evaluation = build_model_evaluation(df)
    selected_name = evaluation["selected_model"]
    selected_eval = evaluation["models"][selected_name]
    threshold = float(selected_eval["threshold"])

    # Fit selected model on 100% of train or entire dataset
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression

    estimators = {
        "Logistic Regression": LogisticRegression(max_iter=1500, class_weight="balanced", random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=8, class_weight="balanced", random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=120, max_depth=2, random_state=42),
    }

    estimator = estimators.get(selected_name, GradientBoostingClassifier(n_estimators=120, max_depth=2, random_state=42))
    pipeline = make_pipeline(estimator)

    X = df[CATEGORICAL + NUMERIC]
    y = df[TARGET].eq("Yes").astype(int)
    pipeline.fit(X, y)

    save_model_artifact(
        pipeline=pipeline,
        model_name=selected_name,
        metrics=evaluation,
        threshold=threshold,
        artifact_dir=artifact_dir,
        version=version,
    )

    return pipeline, {"model_name": selected_name, "threshold": threshold}, evaluation


def get_or_create_model(
    artifact_dir: Path | str | None = None,
    dataset_path: Path | str | None = None,
    fallback_to_training: bool = True,
) -> Tuple[Pipeline, dict[str, Any], dict[str, Any]]:
    """Retrieve persisted model or automatically train and persist if missing."""
    target_dir = Path(artifact_dir) if artifact_dir else CHURN_MODEL_DIR
    try:
        return load_model_artifact(target_dir)
    except (FileNotFoundError, RuntimeError) as exc:
        logger.warning("Could not load artifact from %s: %s", target_dir, exc)
        if fallback_to_training:
            logger.info("Falling back to model training and persistence...")
            return train_and_persist_selected_model(dataset_path=dataset_path, artifact_dir=target_dir)
        raise
