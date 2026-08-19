"""
Agent 5: Predictive Analytics Agent
Models: Prophet (cash flow forecast) + LightGBM (churn) + ARIMA (volume forecast)
Port: 8006
"""
import time
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from fastapi import FastAPI, Depends
from pydantic import BaseModel
import sys
sys.path.append("/app")

from shared.schemas.base import AgentResponse, HealthResponse
from shared.middleware.auth import (
    audit_log_middleware,
    require_secure_configuration,
    verify_service_token,
)
from shared.utils.artefacts import load_verified_artefact
from shared.utils.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()
app = FastAPI(title="SmartBank AI — Predictive Analytics Agent", version="1.0.0")
app.middleware("http")(audit_log_middleware)

churn_model = None
cashflow_model = None
volume_model = None
_start_time = time.time()


@app.on_event("startup")
async def enforce_secure_configuration() -> None:
    """Refuse to serve traffic without a usable service token."""
    require_secure_configuration()


class ChurnRequest(BaseModel):
    customer_id: str
    days_since_last_transaction: int
    monthly_txn_count_trend: float  # positive = growing, negative = declining
    product_count: int
    complaint_count_12m: int
    channel_usage_mobile_pct: float
    account_age_months: int


class CashFlowRequest(BaseModel):
    customer_id: str
    daily_balances: list[float]  # last 90 days, oldest first
    forecast_days: int = 30


class VolumeForecastRequest(BaseModel):
    daily_volumes: list[float]  # last 90 days
    forecast_days: int = 30


@app.on_event("startup")
async def load_models():
    global churn_model, cashflow_model, volume_model
    for filename, variable in [("churn_lgbm.pkl", "churn_model"), ("cashflow_ridge.pkl", "cashflow_model"), ("volume_autoreg_ridge.pkl", "volume_model")]:
        path = Path(settings.model_dir) / filename
        if path.exists():
            globals()[variable] = load_verified_artefact(path)
            logger.info("%s loaded", filename)


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        agent="predictive_analytics", version="1.0.0",
        model_loaded=churn_model is not None and cashflow_model is not None and volume_model is not None,
        uptime_seconds=round(time.time() - _start_time, 1),
    )


@app.post("/predict/churn", response_model=AgentResponse, dependencies=[Depends(verify_service_token)])
async def predict_churn(req: ChurnRequest):
    start = time.monotonic()
    features = pd.DataFrame([{
        "days_since_last_transaction": req.days_since_last_transaction,
        "monthly_transaction_count_trend": req.monthly_txn_count_trend,
        "product_count": req.product_count,
        "complaint_count_12m": req.complaint_count_12m,
        "channel_usage_score": req.channel_usage_mobile_pct,
    }])

    if churn_model is not None:
        trained_model = churn_model["model"] if isinstance(churn_model, dict) else churn_model
        columns = churn_model.get("feature_columns", features.columns.tolist()) if isinstance(churn_model, dict) else features.columns.tolist()
        churn_prob = float(trained_model.predict_proba(features.reindex(columns=columns, fill_value=0))[0][1])
    else:
        # Stub
        churn_prob = min(0.99, max(0.01,
            req.days_since_last_transaction / 180
            - req.product_count * 0.1
            + req.complaint_count_12m * 0.15
        ))

    risk = "HIGH" if churn_prob >= 0.7 else "MEDIUM" if churn_prob >= 0.4 else "LOW"
    latency = (time.monotonic() - start) * 1000
    return AgentResponse(
        agent="predictive_analytics", version="1.0.0", latency_ms=round(latency, 2),
        payload={
            "customer_id": req.customer_id,
            "churn_probability": round(churn_prob, 4),
            "churn_risk": risk,
            "recommended_intervention": (
                "Immediate outreach — personalised retention offer" if risk == "HIGH"
                else "Proactive engagement — product recommendation" if risk == "MEDIUM"
                else "Standard engagement"
            ),
            "human_review_required": True,
            "synthetic_model": churn_model is not None,
        },
    )


@app.post("/predict/cashflow", response_model=AgentResponse, dependencies=[Depends(verify_service_token)])
async def predict_cashflow(req: CashFlowRequest):
    """Use the trained synthetic cash-flow baseline with a safe moving-average fallback."""
    start = time.monotonic()
    balances = req.daily_balances[-30:]
    avg = float(np.mean(balances))
    trend = float(np.polyfit(range(len(balances)), balances, 1)[0])
    if cashflow_model is not None and len(balances) >= 30:
        trained_model = cashflow_model["model"] if isinstance(cashflow_model, dict) else cashflow_model
        values = list(map(float, balances))
        forecast = []
        for step in range(req.forecast_days):
            row = pd.DataFrame([{
                "balance_lag_1": values[-1], "balance_mean_7": float(np.mean(values[-7:])),
                "balance_mean_30": float(np.mean(values[-30:])), "balance_trend_7": values[-1] - values[-8],
                "day_of_week": step % 7,
            }])
            prediction = max(0.0, float(trained_model.predict(row)[0]))
            forecast.append(round(prediction, 2))
            values.append(prediction)
    else:
        forecast = [round(avg + trend * i, 2) for i in range(1, req.forecast_days + 1)]
    lower = [round(f * 0.85, 2) for f in forecast]
    upper = [round(f * 1.15, 2) for f in forecast]

    latency = (time.monotonic() - start) * 1000
    return AgentResponse(
        agent="predictive_analytics", version="1.0.0", latency_ms=round(latency, 2),
        payload={
            "customer_id": req.customer_id,
            "forecast_days": req.forecast_days,
            "forecast": forecast,
            "lower_bound": lower,
            "upper_bound": upper,
            "trend": "increasing" if trend > 0 else "decreasing",
            "human_review_required": True,
            "synthetic_model": cashflow_model is not None,
        },
    )


@app.post("/predict/volume", response_model=AgentResponse, dependencies=[Depends(verify_service_token)])
async def predict_volume(req: VolumeForecastRequest):
    """Use the trained synthetic autoregressive volume baseline with a safe smoothing fallback."""
    start = time.monotonic()
    values = list(map(float, req.daily_volumes))
    trend = float(np.polyfit(range(len(values[-14:])), values[-14:], 1)[0])
    if volume_model is not None and len(values) >= 30:
        trained_model = volume_model["model"] if isinstance(volume_model, dict) else volume_model
        forecast = []
        for step in range(req.forecast_days):
            row = pd.DataFrame([{"lag_1": values[-1], "lag_7": values[-7], "lag_30": values[-30], "day_of_week": step % 7}])
            prediction = max(0.0, float(trained_model.predict(row)[0]))
            forecast.append(round(prediction, 0))
            values.append(prediction)
    else:
        alpha = 0.3
        smoothed = [values[0]]
        for value in values[1:]:
            smoothed.append(alpha * value + (1 - alpha) * smoothed[-1])
        forecast = [round(smoothed[-1] + trend * index, 0) for index in range(1, req.forecast_days + 1)]

    latency = (time.monotonic() - start) * 1000
    return AgentResponse(
        agent="predictive_analytics", version="1.0.0", latency_ms=round(latency, 2),
        payload={
            "forecast_days": req.forecast_days,
            "forecast_volumes": forecast,
            "trend_direction": "increasing" if trend > 0 else "decreasing",
            "trend_daily_change": round(trend, 2),
            "human_review_required": True,
            "synthetic_model": volume_model is not None,
        },
    )
