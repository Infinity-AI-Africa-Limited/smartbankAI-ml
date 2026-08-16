"""
Agent 7: Smart Dashboard Agent
Components: K-Means customer segmentation + NLG insight generator
Port: 8008
"""
import time
import pickle
import logging
import pandas as pd
from pathlib import Path
from fastapi import FastAPI, Depends
from pydantic import BaseModel
import sys
sys.path.append("/app")

from shared.schemas.base import AgentResponse, HealthResponse
from shared.middleware.auth import verify_service_token, audit_log_middleware
from shared.utils.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()
app = FastAPI(title="SmartBank AI — Smart Dashboard Agent", version="1.0.0")
app.middleware("http")(audit_log_middleware)

kmeans_model = None
_start_time = time.time()


class InsightRequest(BaseModel):
    period_label: str  # e.g. "this week"
    txn_volume_current: int
    txn_volume_previous: int
    total_value_ngn: float
    fraud_alerts_count: int
    new_customers: int
    churn_risk_count: int
    top_channel: str
    aml_flags: int


class SegmentRequest(BaseModel):
    customers: list[dict]  # each: {customer_id, days_since_txn, product_count, monthly_txn_avg, account_age_months, income_band_enc}


@app.on_event("startup")
async def load_model():
    global kmeans_model
    path = Path(settings.model_dir) / "dashboard_kmeans.pkl"
    if path.exists():
        with open(path, "rb") as f:
            kmeans_model = pickle.load(f)
        logger.info("K-Means segmentation model loaded")


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        agent="smart_dashboard", version="1.0.0",
        model_loaded=kmeans_model is not None, uptime_seconds=round(time.time() - _start_time, 1),
    )


@app.post("/insights", response_model=AgentResponse, dependencies=[Depends(verify_service_token)])
async def generate_insights(req: InsightRequest):
    start = time.monotonic()
    vol_change_pct = round((req.txn_volume_current - req.txn_volume_previous) / max(req.txn_volume_previous, 1) * 100, 1)
    direction = "up" if vol_change_pct > 0 else "down"

    sentences = []
    sentences.append(
        f"Transaction volume is {direction} {abs(vol_change_pct)}% {req.period_label}, "
        f"driven by {req.top_channel} channel activity "
        f"({req.txn_volume_current:,} transactions totalling ₦{req.total_value_ngn:,.0f})."
    )

    if req.churn_risk_count > 0:
        sentences.append(
            f"{req.churn_risk_count} customer{'s' if req.churn_risk_count > 1 else ''} "
            f"show{'s' if req.churn_risk_count == 1 else ''} early churn signals — "
            f"proactive outreach recommended."
        )

    if req.fraud_alerts_count > 0:
        sentences.append(
            f"Fraud detection flagged {req.fraud_alerts_count} suspicious "
            f"transaction{'s' if req.fraud_alerts_count > 1 else ''}"
            + (f"; {req.aml_flags} AML flag{'s' if req.aml_flags > 1 else ''} pending compliance review." if req.aml_flags > 0 else ", all pending review.")
        )

    if req.new_customers > 0:
        sentences.append(f"{req.new_customers} new customer{'s' if req.new_customers > 1 else ''} onboarded {req.period_label}.")

    narrative = " ".join(sentences)
    latency = (time.monotonic() - start) * 1000
    return AgentResponse(
        agent="smart_dashboard", version="1.0.0", latency_ms=round(latency, 2),
        payload={"narrative": narrative, "metrics_summary": req.model_dump(), "human_review_required": True, "synthetic_model": kmeans_model is not None},
    )


@app.post("/segment", response_model=AgentResponse, dependencies=[Depends(verify_service_token)])
async def segment_customers(req: SegmentRequest):
    start = time.monotonic()
    SEGMENT_LABELS = {0: "High Value", 1: "Growing", 2: "At Risk", 3: "Dormant", 4: "New"}

    results = []
    for c in req.customers:
        if kmeans_model is not None and isinstance(kmeans_model, dict):
            income = {0: 115_000, 1: 285_000, 2: 640_000, 3: 1_450_000}.get(c.get("income_band_enc", 1), 285_000)
            features = pd.DataFrame([{
                "monthly_income_ngn": income,
                "avg_monthly_balance_ngn": income * 1.2,
                "account_age_months": c.get("account_age_months", 12),
                "products_held_count": c.get("product_count", 1),
                "days_since_last_transaction": c.get("days_since_txn", 30),
                "monthly_transaction_count_trend": 0.0,
                "complaint_count_12m": 0,
            }])
            scaled = kmeans_model["scaler"].transform(features[kmeans_model["feature_columns"]])
            seg_id = int(kmeans_model["model"].predict(scaled)[0])
        else:
            # Stub: rule-based segmentation
            days = c.get("days_since_txn", 30)
            products = c.get("product_count", 1)
            seg_id = 3 if days > 90 else 4 if c.get("account_age_months", 12) < 3 else 2 if days > 30 else 1 if products < 2 else 0

        results.append({"customer_id": c["customer_id"], "segment": SEGMENT_LABELS.get(seg_id, "Growing"), "segment_id": seg_id})

    latency = (time.monotonic() - start) * 1000
    return AgentResponse(
        agent="smart_dashboard", version="1.0.0", latency_ms=round(latency, 2),
        payload={"segmented_customers": results, "total": len(results), "human_review_required": True, "synthetic_model": kmeans_model is not None},
    )
