"""
Agent 4: Personalization Agent
Models: Collaborative filtering (SVD) + Next-Best-Action (LightGBM) + K-Means segmentation
Port: 8005
"""
import time
import pickle
import logging
import numpy as np
from pathlib import Path
from fastapi import FastAPI, Depends
import sys
sys.path.append("/app")

from shared.schemas.base import CustomerProfileRequest, AgentResponse, HealthResponse
from shared.middleware.auth import verify_service_token, audit_log_middleware
from shared.utils.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()
app = FastAPI(title="SmartBank AI — Personalization Agent", version="1.0.0")
app.middleware("http")(audit_log_middleware)

nba_model = None
kmeans_model = None
_start_time = time.time()

PRODUCTS = ["savings_account", "fixed_deposit", "personal_loan", "credit_card", "investment_plan", "insurance"]
SEGMENTS = {0: "High Value", 1: "Growing", 2: "At Risk", 3: "Dormant", 4: "New"}
INCOME_MAP = {"low": 0, "mid": 1, "high": 2, "premium": 3}
CHANNEL_MAP = {"mobile": 0, "web": 1, "ussd": 2, "branch": 3}
AGE_MAP = {"18-25": 0, "26-35": 1, "36-45": 2, "46-55": 3, "55+": 4}


@app.on_event("startup")
async def load_models():
    global nba_model, kmeans_model
    model_dir = Path(settings.model_dir)
    for name, var_name in [("nba_lgbm.pkl", "nba_model"), ("kmeans_segments.pkl", "kmeans_model")]:
        path = model_dir / name
        if path.exists():
            with open(path, "rb") as f:
                globals()[var_name] = pickle.load(f)
            logger.info("%s loaded", name)


def featurise(profile: CustomerProfileRequest) -> np.ndarray:
    return np.array([[
        AGE_MAP.get(profile.age_band, 2),
        INCOME_MAP.get(profile.income_band, 1),
        len(profile.products_held),
        CHANNEL_MAP.get(profile.channel_preference, 0),
        profile.days_since_last_transaction,
        profile.monthly_txn_count_3m_avg,
        profile.complaint_count_12m,
        profile.account_age_months,
    ]])


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        agent="personalization", version="1.0.0",
        model_loaded=nba_model is not None,
        uptime_seconds=round(time.time() - _start_time, 1),
    )


@app.post("/recommend", response_model=AgentResponse, dependencies=[Depends(verify_service_token)])
async def recommend(profile: CustomerProfileRequest):
    start = time.monotonic()
    features = featurise(profile)

    if nba_model is not None:
        probs = nba_model.predict_proba(features)[0]
        ranked = sorted(zip(PRODUCTS, probs), key=lambda x: x[1], reverse=True)
    else:
        # Stub: rule-based recommendations
        owned = set(profile.products_held)
        candidates = [p for p in PRODUCTS if p not in owned]
        ranked = [(p, round(0.9 - i * 0.1, 2)) for i, p in enumerate(candidates[:3])]

    segment_id = int(kmeans_model.predict(features)[0]) if kmeans_model is not None else 1
    segment = SEGMENTS.get(segment_id, "Growing")

    top_3 = ranked[:3]
    narrative = (
        f"Based on your profile ({segment} segment), we recommend: "
        + ", ".join(f"{p} ({round(s * 100, 0):.0f}% match)" for p, s in top_3) + "."
    )

    latency = (time.monotonic() - start) * 1000
    return AgentResponse(
        agent="personalization", version="1.0.0", latency_ms=round(latency, 2),
        payload={
            "customer_id": profile.customer_id,
            "segment": segment,
            "recommendations": [{"product": p, "confidence": round(float(s), 4)} for p, s in top_3],
            "next_best_action": top_3[0][0] if top_3 else None,
            "narrative": narrative,
        },
    )
