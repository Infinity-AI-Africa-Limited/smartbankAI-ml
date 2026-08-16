"""
Agent 1: Fraud Detection Agent
Model: LightGBM binary classifier with SHAP explainability
Port: 8002
"""
import time
import logging
import pickle
import json
import numpy as np
from pathlib import Path
from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
import sys
sys.path.append("/app")

from shared.schemas.base import (
    TransactionRequest, AgentResponse, HealthResponse, ExplainResponse, RiskLevel
)
from shared.middleware.auth import verify_service_token, audit_log_middleware
from shared.utils.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()
app = FastAPI(title="SmartBank AI — Fraud Detection Agent", version="1.0.0")

app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)
app.middleware("http")(audit_log_middleware)

# ── Model loading ─────────────────────────────────────────────────────────────

model = None
explainer = None
review_threshold = 0.35
_start_time = time.time()

@app.on_event("startup")
async def load_model():
    global model, explainer, review_threshold
    model_path = Path(settings.model_dir) / "fraud_lgbm.pkl"
    explainer_path = Path(settings.model_dir) / "fraud_shap_explainer.pkl"
    report_path = Path(settings.model_dir) / "evaluation_report.json"

    if model_path.exists():
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        logger.info("Fraud detection model loaded from %s", model_path)
    else:
        logger.warning("Model file not found at %s — running in stub mode", model_path)

    if explainer_path.exists():
        with open(explainer_path, "rb") as f:
            explainer = pickle.load(f)
    if report_path.exists():
        review_threshold = float(json.loads(report_path.read_text()).get("review_threshold", review_threshold))


# ── Feature engineering ───────────────────────────────────────────────────────

CHANNEL_MAP = {"mobile": 0, "web": 1, "ussd": 2, "atm": 3, "pos": 4}

def engineer_features(txn: TransactionRequest) -> np.ndarray:
    channel_enc = CHANNEL_MAP.get(txn.channel.lower(), -1)
    amount_deviation = (
        (txn.amount_ngn - txn.sender_30d_avg_amount) / (txn.sender_30d_avg_amount + 1)
        if txn.sender_30d_avg_amount else 0.0
    )
    is_high_risk_hour = 1 if txn.hour_of_day in range(0, 5) else 0
    is_weekend = 1 if txn.day_of_week in [5, 6] else 0
    velocity = txn.sender_txn_count_1h or 0

    return np.array([[
        txn.amount_ngn,
        channel_enc,
        txn.hour_of_day,
        txn.day_of_week,
        amount_deviation,
        is_high_risk_hour,
        is_weekend,
        velocity,
        txn.sender_30d_avg_amount or 0.0,
    ]])

FEATURE_NAMES = [
    "amount_ngn", "channel_enc", "hour_of_day", "day_of_week",
    "amount_deviation", "is_high_risk_hour", "is_weekend",
    "velocity_1h", "sender_30d_avg"
]


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        agent="fraud_detection",
        version="1.0.0",
        model_loaded=model is not None,
        uptime_seconds=round(time.time() - _start_time, 1),
    )


@app.post("/predict", response_model=AgentResponse, dependencies=[Depends(verify_service_token)])
async def predict(txn: TransactionRequest):
    start = time.monotonic()
    features = engineer_features(txn)

    if model is not None:
        fraud_prob = float(model.predict_proba(features)[0][1])
    else:
        # Stub: rule-based fallback when model not yet trained
        fraud_prob = 0.95 if txn.amount_ngn > 4_900_000 else 0.05

    risk_level = (
        RiskLevel.CRITICAL if fraud_prob >= 0.85 else
        RiskLevel.HIGH if fraud_prob >= 0.65 else
        RiskLevel.MEDIUM if fraud_prob >= 0.35 else
        RiskLevel.LOW
    )

    latency = (time.monotonic() - start) * 1000
    return AgentResponse(
        agent="fraud_detection",
        version="1.0.0",
        latency_ms=round(latency, 2),
        payload={
            "transaction_id": txn.transaction_id,
            "fraud_probability": round(fraud_prob, 4),
            "risk_level": risk_level,
            "action": "REVIEW" if fraud_prob >= review_threshold else "NO_ACTION",
            "human_review_required": True,
            "synthetic_model": model is not None,
        },
    )


@app.post("/explain", response_model=AgentResponse, dependencies=[Depends(verify_service_token)])
async def explain(txn: TransactionRequest):
    features = engineer_features(txn)

    if model is not None:
        fraud_prob = float(model.predict_proba(features)[0][1])
    else:
        fraud_prob = 0.05

    top_factors = []
    if model is not None:
        contributions = model.booster_.predict(features, pred_contrib=True)[0]
        sorted_idx = np.argsort(np.abs(contributions[:-1]))[::-1][:3]
        top_factors = [
            {"feature": FEATURE_NAMES[i], "impact": round(float(contributions[i]), 4)}
            for i in sorted_idx
        ]
    elif explainer is not None:
        shap_values = explainer.shap_values(features)[1][0]
        sorted_idx = np.argsort(np.abs(shap_values))[::-1][:3]
        top_factors = [
            {"feature": FEATURE_NAMES[i], "impact": round(float(shap_values[i]), 4)}
            for i in sorted_idx
        ]
    else:
        top_factors = [
            {"feature": "amount_deviation", "impact": 0.42},
            {"feature": "velocity_1h", "impact": 0.31},
            {"feature": "is_high_risk_hour", "impact": 0.18},
        ]

    risk_level = (
        RiskLevel.CRITICAL if fraud_prob >= 0.85 else
        RiskLevel.HIGH if fraud_prob >= 0.65 else
        RiskLevel.MEDIUM if fraud_prob >= 0.35 else
        RiskLevel.LOW
    )

    narrative = (
        f"This transaction has a development-model fraud probability of {round(fraud_prob * 100, 1)}%. "
        f"The top contributing factors are: "
        + ", ".join(f"{f['feature']} (impact: {f['impact']})" for f in top_factors) + ". Human review is required."
    )

    return AgentResponse(
        agent="fraud_detection",
        version="1.0.0",
        payload=ExplainResponse(
            score=round(fraud_prob, 4),
            risk_level=risk_level,
            top_factors=top_factors,
            narrative=narrative,
        ).model_dump(),
    )
