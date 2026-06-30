"""
Agent 5: Predictive Analytics Agent
Models: Prophet (cash flow forecast) + LightGBM (churn) + ARIMA (volume forecast)
Port: 8006
"""
import time
import pickle
import logging
import numpy as np
from pathlib import Path
from fastapi import FastAPI, Depends
from pydantic import BaseModel
from typing import Optional
import sys
sys.path.append("/app")

from shared.schemas.base import AgentResponse, HealthResponse
from shared.middleware.auth import verify_service_token, audit_log_middleware
from shared.utils.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()
app = FastAPI(title="SmartBank AI — Predictive Analytics Agent", version="1.0.0")
app.middleware("http")(audit_log_middleware)

churn_model = None
_start_time = time.time()


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
    global churn_model
    path = Path(settings.model_dir) / "churn_lgbm.pkl"
    if path.exists():
        with open(path, "rb") as f:
            churn_model = pickle.load(f)
        logger.info("Churn model loaded")


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        agent="predictive_analytics", version="1.0.0",
        model_loaded=churn_model is not None,
        uptime_seconds=round(time.time() - _start_time, 1),
    )


@app.post("/predict/churn", response_model=AgentResponse, dependencies=[Depends(verify_service_token)])
async def predict_churn(req: ChurnRequest):
    start = time.monotonic()
    features = np.array([[
        req.days_since_last_transaction,
        req.monthly_txn_count_trend,
        req.product_count,
        req.complaint_count_12m,
        req.channel_usage_mobile_pct,
        req.account_age_months,
    ]])

    if churn_model is not None:
        churn_prob = float(churn_model.predict_proba(features)[0][1])
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
        },
    )


@app.post("/predict/cashflow", response_model=AgentResponse, dependencies=[Depends(verify_service_token)])
async def predict_cashflow(req: CashFlowRequest):
    """Simple moving-average forecast — replace with Prophet in production."""
    start = time.monotonic()
    balances = req.daily_balances[-30:]  # use last 30 days
    avg = float(np.mean(balances))
    trend = float(np.polyfit(range(len(balances)), balances, 1)[0])

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
        },
    )


@app.post("/predict/volume", response_model=AgentResponse, dependencies=[Depends(verify_service_token)])
async def predict_volume(req: VolumeForecastRequest):
    """Simple exponential smoothing forecast — replace with ARIMA in production."""
    start = time.monotonic()
    alpha = 0.3
    smoothed = [req.daily_volumes[0]]
    for v in req.daily_volumes[1:]:
        smoothed.append(alpha * v + (1 - alpha) * smoothed[-1])

    last = smoothed[-1]
    trend = float(np.polyfit(range(len(smoothed[-14:])), smoothed[-14:], 1)[0])
    forecast = [round(last + trend * i, 0) for i in range(1, req.forecast_days + 1)]

    latency = (time.monotonic() - start) * 1000
    return AgentResponse(
        agent="predictive_analytics", version="1.0.0", latency_ms=round(latency, 2),
        payload={
            "forecast_days": req.forecast_days,
            "forecast_volumes": forecast,
            "trend_direction": "increasing" if trend > 0 else "decreasing",
            "trend_daily_change": round(trend, 2),
        },
    )
