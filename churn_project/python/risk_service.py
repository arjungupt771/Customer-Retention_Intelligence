"""Customer risk segmentation, aggregation, and action center service."""
from __future__ import annotations

import io
import logging
from typing import Any, Mapping

import numpy as np
import pandas as pd

from churn_project.python.churn_analysis import CATEGORICAL, NUMERIC, TARGET
from churn_project.python.prediction_service import PredictionService, categorize_risk

logger = logging.getLogger(__name__)

DEFAULT_RISK_CONFIG = {
    "high": {"min_probability": 0.70},
    "medium": {"min_probability": 0.30},
    "low": {"min_probability": 0.00},
}


class RiskSegmentationService:
    """Service to score customer population, aggregate risk, and power retention actions."""

    def __init__(
        self,
        prediction_service: PredictionService,
        df: pd.DataFrame,
        risk_config: dict[str, dict[str, float]] | None = None,
    ) -> None:
        self.prediction_service = prediction_service
        self.risk_config = risk_config or DEFAULT_RISK_CONFIG
        self._scored_df: pd.DataFrame | None = None
        self._init_scored_data(df)

    def _init_scored_data(self, df: pd.DataFrame) -> None:
        """Score customer base once using model pipeline and store enriched DataFrame."""
        pipeline = self.prediction_service.pipeline
        threshold = self.prediction_service.threshold

        features = df[CATEGORICAL + NUMERIC]
        probabilities = pipeline.predict_proba(features)[:, 1]

        high_min = self.risk_config.get("high", {}).get("min_probability", 0.70)
        med_min = self.risk_config.get("medium", {}).get("min_probability", 0.30)

        # Vectorized risk categorization
        conditions = [probabilities >= high_min, probabilities >= med_min]
        choices = ["High", "Medium"]
        risk_categories = np.select(conditions, choices, default="Low")

        scored = df.copy()
        scored["churn_probability"] = np.round(probabilities, 4)
        scored["predicted_churn"] = probabilities >= threshold
        scored["risk_category"] = risk_categories
        self._scored_df = scored

    @property
    def scored_customers(self) -> pd.DataFrame:
        if self._scored_df is None:
            raise RuntimeError("Scored customer data is not initialized")
        return self._scored_df

    def get_risk_distribution(self) -> dict[str, dict[str, Any]]:
        """Return counts and percentages across High, Medium, and Low risk."""
        df = self.scored_customers
        total = len(df)
        if total == 0:
            return {
                "high": {"count": 0, "percentage": 0.0},
                "medium": {"count": 0, "percentage": 0.0},
                "low": {"count": 0, "percentage": 0.0},
            }

        high_count = int(df.risk_category.eq("High").sum())
        med_count = int(df.risk_category.eq("Medium").sum())
        low_count = int(df.risk_category.eq("Low").sum())

        return {
            "high": {
                "count": high_count,
                "percentage": round((high_count / total) * 100, 1),
            },
            "medium": {
                "count": med_count,
                "percentage": round((med_count / total) * 100, 1),
            },
            "low": {
                "count": low_count,
                "percentage": round((low_count / total) * 100, 1),
            },
        }

    def get_action_center_summary(self) -> dict[str, Any]:
        """Calculate executive summary metrics for retention action center."""
        df = self.scored_customers
        total = len(df)
        dist = self.get_risk_distribution()
        high_df = df[df.risk_category.eq("High")]

        avg_prob = float(df.churn_probability.mean()) if total else 0.0
        predicted_churn_rate = float(df.predicted_churn.mean() * 100) if total else 0.0
        revenue_at_risk = float(high_df.MonthlyCharges.sum()) if len(high_df) else 0.0

        return {
            "total_customers": total,
            "high_risk_customers": dist["high"]["count"],
            "high_risk_percentage": dist["high"]["percentage"],
            "medium_risk_customers": dist["medium"]["count"],
            "low_risk_customers": dist["low"]["count"],
            "average_churn_probability": round(avg_prob, 4),
            "predicted_churn_rate": round(predicted_churn_rate, 2),
            "monthly_revenue_at_risk": round(revenue_at_risk, 2),
            "risk_distribution": dist,
        }

    def filter_customers(
        self,
        risk_category: str | None = None,
        contract: str | None = None,
        payment_method: str | None = None,
        min_tenure: int | None = None,
        max_tenure: int | None = None,
        min_monthly_charges: float | None = None,
        max_monthly_charges: float | None = None,
        search: str | None = None,
        sort_by: str = "churn_probability",
        ascending: bool = False,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[dict[str, Any]], int]:
        """Filter, rank, and paginate customer retention targets."""
        df = self.scored_customers

        if risk_category and risk_category.lower() != "all":
            df = df[df.risk_category.str.lower().eq(risk_category.lower())]
        if contract and contract.lower() != "all":
            df = df[df.Contract.eq(contract)]
        if payment_method and payment_method.lower() != "all":
            df = df[df.PaymentMethod.eq(payment_method)]
        if min_tenure is not None:
            df = df[df.tenure.ge(min_tenure)]
        if max_tenure is not None:
            df = df[df.tenure.le(max_tenure)]
        if min_monthly_charges is not None:
            df = df[df.MonthlyCharges.ge(min_monthly_charges)]
        if max_monthly_charges is not None:
            df = df[df.MonthlyCharges.le(max_monthly_charges)]
        if search:
            search_str = str(search).strip().lower()
            df = df[df.customerID.str.lower().str.contains(search_str)]

        total_filtered = len(df)

        if sort_by in df.columns:
            df = df.sort_values(by=sort_by, ascending=ascending)
        else:
            df = df.sort_values(by="churn_probability", ascending=False)

        paginated = df.iloc[offset : offset + limit]
        return paginated.to_dict(orient="records"), total_filtered

    def export_csv(
        self,
        risk_category: str | None = None,
        contract: str | None = None,
        payment_method: str | None = None,
        min_tenure: int | None = None,
        max_tenure: int | None = None,
        min_monthly_charges: float | None = None,
        max_monthly_charges: float | None = None,
        search: str | None = None,
        limit: int = 5000,
    ) -> str:
        """Export filtered customer predictions to CSV string."""
        records, _ = self.filter_customers(
            risk_category=risk_category,
            contract=contract,
            payment_method=payment_method,
            min_tenure=min_tenure,
            max_tenure=max_tenure,
            min_monthly_charges=min_monthly_charges,
            max_monthly_charges=max_monthly_charges,
            search=search,
            limit=limit,
            offset=0,
        )

        df_export = pd.DataFrame(records)
        output = io.StringIO()
        df_export.to_csv(output, index=False)
        return output.getvalue()
