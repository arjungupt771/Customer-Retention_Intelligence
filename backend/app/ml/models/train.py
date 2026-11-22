"""Model instantiation and training module."""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline

from backend.app.config import GLOBAL_CONFIG
from backend.app.ml.data.preprocessing import make_pipeline
from backend.app.ml.data.validation import CATEGORICAL, NUMERIC, TARGET

logger = logging.getLogger(__name__)


def get_model_candidates(config=GLOBAL_CONFIG) -> dict[str, Any]:
    """Instantiate candidate estimators with configured hyperparameters."""
    return {
        "Logistic Regression": LogisticRegression(**config.models.logistic_regression),
        "Random Forest": RandomForestClassifier(**config.models.random_forest),
        "Gradient Boosting": GradientBoostingClassifier(**config.models.gradient_boosting),
    }


def fit_model(estimator, X: pd.DataFrame, y: np.ndarray) -> Pipeline:
    """Fit a pipeline combining preprocessing and estimator."""
    pipeline = make_pipeline(estimator)
    pipeline.fit(X, y)
    return pipeline
