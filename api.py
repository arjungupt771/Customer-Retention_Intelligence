"""Cached, validated API for the telecom churn portfolio dashboard."""
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from churn_project.python.churn_analysis import (
    build_model_evaluation, clean_dataset, find_dataset, key_metrics,
)

app = FastAPI(title="Churn Analysis API", version="2.0.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True, allow_methods=["GET"], allow_headers=["*"],
)


class MetricResponse(BaseModel):
    total_customers: int
    active_customers: int
    churned_customers: int
    churn_rate: float
    active_monthly_revenue: float
    historical_churned_monthly_revenue: float
    active_monthly_revenue_at_risk: float
    revenue_loss_percentage: float
    population: str
    period: str


@lru_cache(maxsize=1)
def load_dataset() -> pd.DataFrame:
    try:
        path = find_dataset(Path(__file__).resolve().parent)
        return clean_dataset(pd.read_csv(path))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except (ValueError, pd.errors.ParserError) as exc:
        raise HTTPException(status_code=422, detail=f"Dataset validation failed: {exc}") from exc


@lru_cache(maxsize=1)
def cached_business_analysis() -> dict[str, Any]:
    df = load_dataset()
    contracts = grouped_metrics(df, "Contract")
    payments = grouped_metrics(df, "PaymentMethod")
    tenure = df.assign(tenure_cohort=pd.cut(
        df.tenure, bins=[-1, 12, 24, 48, 100],
        labels=["0–12 months", "13–24 months", "25–48 months", "49+ months"],
    ))
    cohorts = grouped_metrics(tenure, "tenure_cohort")
    return {
        "data_summary": {
            "total_rows": int(len(df)), "total_columns": int(df.shape[1]),
            "churn_rate": key_metrics(df)["churn_rate"],
            "missing_values": int(df.isna().sum().sum()),
            "unique_customers": int(df.customerID.nunique()),
            "memory_usage": f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB",
            "numeric_columns": int(len(df.select_dtypes(include=[np.number]).columns)),
            "categorical_columns": int(len(df.columns) - len(df.select_dtypes(include=[np.number]).columns)),
            "validation": {"duplicate_customer_ids": 0, "invalid_records": 0},
            "target_distribution": {
                "churned": int(df.Churn.eq("Yes").sum()),
                "active": int(df.Churn.eq("No").sum()),
            },
        },
        "key_metrics": key_metrics(df),
        "contract_analysis": {"data": contracts},
        "payment_analysis": {"data": payments},
        "tenure_analysis": {"data": cohorts},
    }


@lru_cache(maxsize=1)
def cached_model_evaluation() -> dict[str, Any]:
    return build_model_evaluation(load_dataset())


def grouped_metrics(df: pd.DataFrame, column: str) -> list[dict[str, Any]]:
    rows = []
    for name, group in df.groupby(column, observed=True, dropna=False):
        rows.append({
            "name": str(name), "customers": int(len(group)),
            "churned": int(group.Churn.eq("Yes").sum()),
            "churn_rate": round(float(group.Churn.eq("Yes").mean() * 100), 2) if len(group) else 0,
            "monthly_revenue": round(float(group.MonthlyCharges.sum()), 2),
        })
    return rows


@app.get("/")
def root():
    return {"message": "Churn Analysis API", "status": "running"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/ready")
def ready():
    try:
        df = load_dataset()
        return {"status": "ready", "customers": int(len(df))}
    except HTTPException as exc:
        raise HTTPException(status_code=503, detail=exc.detail) from exc


@app.get("/api/data-summary")
def data_summary():
    return cached_business_analysis()["data_summary"]


@app.get("/api/key-metrics", response_model=MetricResponse)
def metrics():
    return cached_business_analysis()["key_metrics"]


@app.get("/api/contract-analysis")
def contract_analysis():
    return cached_business_analysis()["contract_analysis"]


@app.get("/api/payment-analysis")
def payment_analysis():
    return cached_business_analysis()["payment_analysis"]


@app.get("/api/tenure-analysis")
def tenure_analysis():
    return cached_business_analysis()["tenure_analysis"]


@app.get("/api/segments")
def segments(
    contract: str | None = Query(default=None, pattern="^(Month-to-month|One year|Two year)$"),
    tenure_group: str | None = Query(default=None, pattern="^(0–12 months|13–24 months|25–48 months|49\\+ months)$"),
    payment_method: str | None = Query(default=None, pattern="^(Electronic check|Mailed check|Bank transfer \\(automatic\\)|Credit card \\(automatic\\))$"),
    segment: str | None = Query(default=None, pattern="^(Higher heuristic score|Lower heuristic score)$"),
    churn: str | None = Query(default=None, pattern="^(Yes|No)$"),
):
    df = load_dataset()
    if contract:
        df = df[df.Contract.eq(contract)]
    if payment_method:
        df = df[df.PaymentMethod.eq(payment_method)]
    if tenure_group:
        df = df.assign(_tenure_group=pd.cut(
            df.tenure, bins=[-1, 12, 24, 48, 100],
            labels=["0–12 months", "13–24 months", "25–48 months", "49+ months"],
        ))
        df = df[df._tenure_group.eq(tenure_group)]
    if segment:
        if segment not in {"Higher heuristic score", "Lower heuristic score"}:
            raise HTTPException(status_code=422, detail="segment must be a supported heuristic score group")
        score = heuristic_risk_score(df)
        df = df.loc[score.ge(40) if segment == "Higher heuristic score" else score.lt(40)]
    if churn:
        df = df[df.Churn.eq(churn)]
    return {"filters": {"contract": contract, "tenure_group": tenure_group,
                         "payment_method": payment_method, "segment": segment, "churn": churn},
            "metrics": key_metrics(df), "count": int(len(df))}


def heuristic_risk_score(df: pd.DataFrame) -> pd.Series:
    # Documented descriptive score; it is not a model probability.
    return (
        df.Contract.eq("Month-to-month").astype(int) * 30
        + df.InternetService.eq("Fiber optic").astype(int) * 15
        + df.PaymentMethod.eq("Electronic check").astype(int) * 12
        + df.TechSupport.eq("No").astype(int) * 10
        + df.SeniorCitizen.eq(1).astype(int) * 8
        + df.tenure.lt(12).astype(int) * 10
        + df.MonthlyCharges.gt(75).astype(int) * 8
    )


@app.get("/api/ml-models")
def ml_models():
    return cached_model_evaluation()


@app.get("/api/full-analysis")
def full_analysis():
    return {**cached_business_analysis(), "ml_models": cached_model_evaluation()}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=int(__import__("os").environ.get("PORT", 8000)))
