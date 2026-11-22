"""Model persistence and artifact management."""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Tuple

import joblib
from sklearn.pipeline import Pipeline

from backend.app.config import GLOBAL_CONFIG, MODELS_DIR
from backend.app.ml.data.validation import CATEGORICAL, NUMERIC, TARGET

logger = logging.getLogger(__name__)


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
    target_dir = Path(artifact_dir) if artifact_dir else MODELS_DIR
    target_dir.mkdir(parents=True, exist_ok=True)

    model_path = target_dir / "model.joblib"
    joblib.dump(pipeline, model_path, compress=3)

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
            "high": {"min_probability": GLOBAL_CONFIG.risk.high_min_probability},
            "medium": {"min_probability": GLOBAL_CONFIG.risk.medium_min_probability},
            "low": {"min_probability": GLOBAL_CONFIG.risk.low_min_probability},
        },
        "config": GLOBAL_CONFIG.to_dict(),
        "sklearn_version": joblib.__version__,
        **(extra_metadata or {}),
    }
    with open(target_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    with open(target_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    logger.info("Saved model artifact to %s (model: %s, threshold: %.3f)", target_dir, model_name, threshold)
    return target_dir


def load_model_artifact(
    artifact_dir: Path | str | None = None,
) -> Tuple[Pipeline, dict[str, Any], dict[str, Any]]:
    """Load model pipeline, metadata, and metrics from artifact directory."""
    target_dir = Path(artifact_dir) if artifact_dir else MODELS_DIR
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


def persist_selected_threshold(
    new_threshold: float,
    artifact_dir: Path | str | None = None,
) -> dict[str, Any]:
    """Persist updated operating threshold to metadata.json."""
    target_dir = Path(artifact_dir) if artifact_dir else MODELS_DIR
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
