"""Preprocessing pipelines for scikit-learn models."""
from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from backend.app.ml.data.validation import CATEGORICAL, NUMERIC


def make_preprocessor() -> ColumnTransformer:
    """Construct leakage-safe ColumnTransformer for categorical and numeric features."""
    return ColumnTransformer([
        ("categorical", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), CATEGORICAL),
        ("numeric", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), NUMERIC),
    ])


def make_pipeline(model) -> Pipeline:
    """Wrap preprocessor and estimator into a unified Pipeline."""
    return Pipeline([
        ("preprocess", make_preprocessor()),
        ("model", model),
    ])
