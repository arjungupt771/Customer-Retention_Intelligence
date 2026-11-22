"""Tests for Phase 8: ML Pipeline Refactoring & Configuration."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from backend.app.config import GLOBAL_CONFIG, PipelineConfig
from backend.app.ml.data.preprocessing import make_pipeline, make_preprocessor
from backend.app.ml.data.validation import clean_dataset, find_dataset
from backend.app.ml.models.persistence import load_model_artifact, save_model_artifact
from backend.app.ml.models.predict import PredictionService
from backend.app.ml.models.train import get_model_candidates
from backend.app.ml.pipeline import ChurnTrainingPipeline


def test_configuration_management(tmp_path):
    config = PipelineConfig()
    assert config.training.random_seed == 42
    assert config.training.test_size == 0.25
    assert config.costs.missed_churner_cost == 500.0

    # Test serialization
    cfg_file = tmp_path / "config.json"
    config.save_json(cfg_file)
    assert cfg_file.exists()


def test_modular_preprocessing():
    preprocessor = make_preprocessor()
    transformer_names = [t[0] for t in preprocessor.transformers]
    assert "categorical" in transformer_names
    assert "numeric" in transformer_names


def test_unified_training_pipeline_execution(tmp_path):
    # Test pipeline execution on small sample for speed
    csv_file = find_dataset()
    df = clean_dataset(pd.read_csv(csv_file))
    sample = df.groupby("Churn", group_keys=False).sample(n=60, random_state=42).reset_index(drop=True)
    sample_csv = tmp_path / "sample.csv"
    sample.to_csv(sample_csv, index=False)

    pipeline_runner = ChurnTrainingPipeline()
    result = pipeline_runner.run(dataset_path=sample_csv, artifact_dir=tmp_path)

    assert result["status"] == "success"
    assert "selected_model" in result
    assert "threshold" in result
    assert "execution_time_seconds" in result
    assert (tmp_path / "model.joblib").exists()
    assert (tmp_path / "metadata.json").exists()
    assert (tmp_path / "metrics.json").exists()

    # Load and verify prediction works with newly persisted pipeline
    loaded_pipeline, meta, _ = load_model_artifact(tmp_path)
    pred_svc = PredictionService(pipeline=loaded_pipeline, metadata=meta)
    res = pred_svc.predict_one({"gender": "Female", "tenure": 12, "MonthlyCharges": 70.0})
    assert "churn_probability" in res
