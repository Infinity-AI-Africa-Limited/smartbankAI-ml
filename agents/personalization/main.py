"""
Agent 4: Personalization Agent
Models: Collaborative filtering (SVD) + Next-Best-Action (LightGBM) + K-Means segmentation
Port: 8005
"""
import time
import pickle
import logging
import numpy as np
import pandas as pd
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
product_similarity = None
_start_time = time.time()

PRODUCTS = ["savings_account", "fixed_deposit", "personal_loan", "credit_card", "investment_plan", "insurance"]
SEGMENTS = {0: "High Value", 1: "Growing", 2: "At Risk", 3: "Dormant", 4: "New"}
INCOME_MAP = {"low": 0, "mid": 1, "high": 2, "premium": 3}
CHANNEL_MAP = {"mobile": 0, "web": 1, "ussd": 2, "branch": 3}
AGE_MAP = {"18-25": 0, "26-35": 1, "36-45": 2, "46-55": 3, "55+": 4}


@app.on_event("startup")
async def load_models():
    global nba_model, kmeans_model, product_similarity
    model_dir = Path(settings.model_dir)
    for name, var_name in [("nba_lgbm.pkl", "nba_model"), ("kmeans_segments.pkl", "kmeans_model")]:
        path = model_dir / name
        if path.exists():
            with open(path, "rb") as f:
                globals()[var_name] = pickle.load(f)
            logger.info("%s loaded", name)
    similarity_path = model_dir / "product_similarity.csv"
    if similarity_path.exists():
        product_similarity = pd.read_csv(similarity_path, index_col=0)


def model_frame(profile: CustomerProfileRequest) -> pd.DataFrame:
    age_by_band = {"18-25": 22, "26-35": 30, "36-50": 43, "51+": 56, "46-55": 50, "55+": 58}
    income_by_band = {"low": 115_000, "mid": 285_000, "upper_mid": 640_000, "high": 1_450_000, "premium": 1_450_000}
    income = income_by_band.get(profile.income_band, 285_000)
    return pd.DataFrame([{
        "age": age_by_band.get(profile.age_band, 35), "monthly_income_ngn": income,
        "account_age_months": profile.account_age_months, "products_held_count": len(profile.products_held),
        "channel_preference": profile.channel_preference, "income_band": profile.income_band,
        "avg_monthly_balance_ngn": income * 1.2,
    }])


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
    features = model_frame(profile)

    if nba_model is not None:
        encoded = pd.get_dummies(features[["age", "monthly_income_ngn", "account_age_months", "products_held_count", "channel_preference", "income_band"]], dtype=float)
        model_object = nba_model["model"] if isinstance(nba_model, dict) else nba_model
        columns = nba_model.get("columns", encoded.columns.tolist()) if isinstance(nba_model, dict) else encoded.columns.tolist()
        encoded = encoded.reindex(columns=columns, fill_value=0)
        classes = nba_model.get("classes", PRODUCTS) if isinstance(nba_model, dict) else PRODUCTS
        probabilities = dict(zip(classes, model_object.predict_proba(encoded)[0]))
        owned = set(profile.products_held)
        similarity_scores = {}
        if product_similarity is not None and owned:
            for product in PRODUCTS:
                related = [float(product_similarity.loc[item, product]) for item in owned if item in product_similarity.index and product in product_similarity.columns]
                similarity_scores[product] = float(np.mean(related)) if related else 0.0
        ranked = [
            (product, 0.65 * float(probabilities.get(product, 0.0)) + 0.35 * max(0.0, similarity_scores.get(product, 0.0)))
            for product in PRODUCTS if product not in owned
        ]
        ranked.sort(key=lambda item: item[1], reverse=True)
    else:
        # Stub: rule-based recommendations
        owned = set(profile.products_held)
        candidates = [p for p in PRODUCTS if p not in owned]
        ranked = [(p, round(0.9 - i * 0.1, 2)) for i, p in enumerate(candidates[:3])]

    if kmeans_model is not None and isinstance(kmeans_model, dict):
        segment_values = features[kmeans_model["feature_columns"]]
        segment_id = int(kmeans_model["model"].predict(kmeans_model["scaler"].transform(segment_values))[0])
        segment = kmeans_model.get("segment_names", {}).get(segment_id, "Growing")
    else:
        segment = "Growing"

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
            "human_review_required": True,
            "synthetic_model": nba_model is not None,
        },
    )
