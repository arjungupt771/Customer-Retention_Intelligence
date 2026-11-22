"""Statistical bootstrap confidence intervals and model comparison."""
from __future__ import annotations

from churn_project.python.confidence_service import (
    DEFAULT_BOOTSTRAP_CONFIG,
    compare_models_bootstrap,
    compute_bootstrap_confidence_intervals,
)

__all__ = [
    "compute_bootstrap_confidence_intervals",
    "compare_models_bootstrap",
    "DEFAULT_BOOTSTRAP_CONFIG",
]
