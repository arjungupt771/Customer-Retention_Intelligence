"""SHAP-based model explainability service for global and local customer predictions."""
from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd
import shap
from sklearn.pipeline import Pipeline

from churn_project.python.churn_analysis import CATEGORICAL, NUMERIC, TARGET

logger = logging.getLogger(__name__)


def clean_feature_name(name: str) -> str:
    """Strip pipeline prefixes like categorical__ or numeric__."""
    return name.replace("categorical__", "").replace("numeric__", "")


def parse_feature_info(raw_name: str) -> tuple[str, str | None]:
    """Extract original column name and specific category if one-hot encoded."""
    cleaned = clean_feature_name(raw_name)
    for cat in CATEGORICAL:
        prefix = f"{cat}_"
        if cleaned.startswith(prefix):
            val = cleaned[len(prefix):]
            return cat, val
    for num in NUMERIC:
        if cleaned == num:
            return num, None
    return cleaned, None


class ModelExplainabilityService:
    """SHAP explainer for Gradient Boosting / tree-based churn models."""

    def __init__(self, pipeline: Pipeline, background_df: pd.DataFrame | None = None) -> None:
        self.pipeline = pipeline
        self.preprocessor = pipeline.named_steps["preprocess"]
        self.model = pipeline.named_steps["model"]

        self.transformed_feature_names: list[str] = [
            str(f) for f in self.preprocessor.get_feature_names_out()
        ]
        self.clean_feature_names: list[str] = [
            clean_feature_name(f) for f in self.transformed_feature_names
        ]

        # Initialize TreeExplainer
        self.explainer = shap.TreeExplainer(self.model)
        self.expected_value = float(
            self.explainer.expected_value[0]
            if isinstance(self.explainer.expected_value, (list, np.ndarray))
            else self.explainer.expected_value
        )
        self._global_importance_cache: list[dict[str, Any]] | None = None

    def explain_instance(
        self,
        customer_row: pd.Series | dict[str, Any],
        top_n: int = 5,
    ) -> dict[str, Any]:
        """Explain an individual customer's prediction using SHAP."""
        if isinstance(customer_row, dict):
            row_df = pd.DataFrame([customer_row])
        elif isinstance(customer_row, pd.Series):
            row_df = pd.DataFrame([customer_row.to_dict()])
        else:
            row_df = customer_row

        X_input = row_df[CATEGORICAL + NUMERIC]
        X_trans = self.preprocessor.transform(X_input)

        shap_vals = self.explainer.shap_values(X_trans)
        if isinstance(shap_vals, list):
            # For some tree classifiers, shap_values is a list of [neg_class, pos_class]
            shap_row = shap_vals[1][0] if len(shap_vals) > 1 else shap_vals[0][0]
        elif len(shap_vals.shape) == 3:
            shap_row = shap_vals[0, :, 1]
        else:
            shap_row = shap_vals[0]

        # Predict probability
        prob = float(self.pipeline.predict_proba(X_input)[0, 1])

        # Aggregate SHAP contributions by original feature
        feature_contributions: list[dict[str, Any]] = []
        original_feature_totals: dict[str, float] = {feat: 0.0 for feat in (CATEGORICAL + NUMERIC)}
        original_feature_details: dict[str, list[str]] = {feat: [] for feat in (CATEGORICAL + NUMERIC)}

        for raw_name, shap_val in zip(self.transformed_feature_names, shap_row):
            orig_col, cat_val = parse_feature_info(raw_name)
            val = float(shap_val)
            original_feature_totals[orig_col] = original_feature_totals.get(orig_col, 0.0) + val

            display_val = row_df[orig_col].iloc[0] if orig_col in row_df.columns else cat_val
            feature_contributions.append({
                "transformed_feature": clean_feature_name(raw_name),
                "original_feature": orig_col,
                "feature_value": str(display_val),
                "shap_value": round(val, 4),
                "impact": "risk" if val > 0 else "protective",
            })

        # Rank original features by aggregated SHAP contribution
        sorted_features = sorted(
            original_feature_totals.items(), key=lambda item: abs(item[1]), reverse=True
        )

        top_risk_factors = []
        top_protective_factors = []

        for feat, total_shap in sorted_features:
            val = row_df[feat].iloc[0] if feat in row_df.columns else "N/A"
            item = {
                "feature": feat,
                "value": str(val),
                "contribution": round(float(total_shap), 4),
                "description": self._format_factor_description(feat, val, total_shap),
            }
            if total_shap > 0.005:
                top_risk_factors.append(item)
            elif total_shap < -0.005:
                top_protective_factors.append(item)

        return {
            "churn_probability": round(prob, 2),
            "predicted_churn": bool(prob >= 0.5),
            "base_value": round(self.expected_value, 4),
            "feature_contributions": sorted(feature_contributions, key=lambda x: abs(x["shap_value"]), reverse=True)[:15],
            "top_risk_factors": top_risk_factors[:top_n],
            "top_protective_factors": top_protective_factors[:top_n],
            "summary_text": self._generate_summary_text(prob, top_risk_factors[:3], top_protective_factors[:3]),
        }

    def explain_global(self, df_sample: pd.DataFrame, max_samples: int = 500) -> list[dict[str, Any]]:
        """Calculate global feature importance ranking via mean absolute SHAP values."""
        if self._global_importance_cache is not None:
            return self._global_importance_cache

        sample = df_sample.sample(n=min(len(df_sample), max_samples), random_state=42)
        X_trans = self.preprocessor.transform(sample[CATEGORICAL + NUMERIC])

        shap_vals = self.explainer.shap_values(X_trans)
        if isinstance(shap_vals, list):
            sv = shap_vals[1] if len(shap_vals) > 1 else shap_vals[0]
        elif len(shap_vals.shape) == 3:
            sv = shap_vals[:, :, 1]
        else:
            sv = shap_vals

        mean_abs_shap = np.mean(np.abs(sv), axis=0)

        # Aggregate across original features
        orig_importance: dict[str, float] = {}
        for raw_name, imp in zip(self.transformed_feature_names, mean_abs_shap):
            orig_col, _ = parse_feature_info(raw_name)
            orig_importance[orig_col] = orig_importance.get(orig_col, 0.0) + float(imp)

        total_imp = sum(orig_importance.values()) or 1.0
        results = [
            {
                "feature": feat,
                "mean_abs_shap": round(float(val), 4),
                "relative_importance_pct": round((float(val) / total_imp) * 100, 1),
            }
            for feat, val in sorted(orig_importance.items(), key=lambda item: item[1], reverse=True)
        ]
        self._global_importance_cache = results
        return results

    def _format_factor_description(self, feature: str, value: Any, shap_val: float) -> str:
        """Create human-readable bullet description."""
        direction = "increases churn risk" if shap_val > 0 else "reduces churn risk"
        if feature == "Contract":
            return f"{value} contract ({direction})"
        if feature == "tenure":
            return f"Tenure of {value} months ({direction})"
        if feature == "MonthlyCharges":
            return f"Monthly charge of ${value} ({direction})"
        if feature == "PaymentMethod":
            return f"{value} payment ({direction})"
        if feature == "InternetService":
            return f"{value} internet service ({direction})"
        return f"{feature} = {value} ({direction})"

    def _generate_summary_text(
        self,
        prob: float,
        risk_factors: list[dict[str, Any]],
        protective_factors: list[dict[str, Any]],
    ) -> str:
        lines = [f"Customer Churn Probability: {int(round(prob * 100))}%"]
        if risk_factors:
            lines.append("Factors Increasing Churn Risk:")
            for rf in risk_factors:
                lines.append(f"• {rf['description']}")
        if protective_factors:
            lines.append("Factors Reducing Churn Risk:")
            for pf in protective_factors:
                lines.append(f"• {pf['description']}")
        return "\n".join(lines)
