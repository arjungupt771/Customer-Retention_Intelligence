"""Centralized configuration management for Churn Intelligence ML pipeline."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS_ROOT = PROJECT_ROOT / "artifacts"
MODELS_DIR = ARTIFACTS_ROOT / "models" / "churn_model"
MONITORING_DIR = ARTIFACTS_ROOT / "monitoring"


@dataclass(frozen=True)
class DataConfig:
    target: str = "Churn"
    categorical_columns: tuple[str, ...] = (
        "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
        "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
        "PaperlessBilling", "PaymentMethod",
    )
    numeric_columns: tuple[str, ...] = (
        "SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges",
    )


@dataclass(frozen=True)
class TrainingConfig:
    random_seed: int = 42
    test_size: float = 0.25
    cv_splits: int = 5
    validation_split: float = 0.20


@dataclass(frozen=True)
class ModelParamsConfig:
    gradient_boosting: dict[str, Any] = field(
        default_factory=lambda: {
            "n_estimators": 120,
            "max_depth": 2,
            "random_state": 42,
        }
    )
    random_forest: dict[str, Any] = field(
        default_factory=lambda: {
            "n_estimators": 200,
            "max_depth": 8,
            "class_weight": "balanced",
            "random_state": 42,
            "n_jobs": -1,
        }
    )
    logistic_regression: dict[str, Any] = field(
        default_factory=lambda: {
            "max_iter": 1500,
            "class_weight": "balanced",
            "random_state": 42,
        }
    )


@dataclass(frozen=True)
class CostConfig:
    missed_churner_cost: float = 500.0
    false_positive_cost: float = 50.0
    intervention_cost: float = 25.0


@dataclass(frozen=True)
class RiskCategoryConfig:
    high_min_probability: float = 0.70
    medium_min_probability: float = 0.30
    low_min_probability: float = 0.00


@dataclass(frozen=True)
class DriftConfig:
    low_threshold: float = 0.10
    moderate_threshold: float = 0.20
    high_threshold: float = 0.30


@dataclass(frozen=True)
class BootstrapConfig:
    n_bootstraps: int = 500
    confidence_level: float = 0.95
    random_seed: int = 42


@dataclass(frozen=True)
class PipelineConfig:
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    models: ModelParamsConfig = field(default_factory=ModelParamsConfig)
    costs: CostConfig = field(default_factory=CostConfig)
    risk: RiskCategoryConfig = field(default_factory=RiskCategoryConfig)
    drift: DriftConfig = field(default_factory=DriftConfig)
    bootstrap: BootstrapConfig = field(default_factory=BootstrapConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save_json(self, path: Path | str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)


GLOBAL_CONFIG = PipelineConfig()
