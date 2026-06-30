"""
Agent 2: Credit Risk Agent
Models: WoE Logistic Regression Scorecard (CBN-explainable) + LightGBM challenger
Port: 8003
"""
import time
import pickle
import logging
import numpy as np
from pathlib import Path
from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
import sys
sys.path.append("/app")

from shared.schemas.base import LoanApplicationRequest, AgentResponse, HealthResponse, RiskLevel
from shared.middleware.auth import verify_service_token, audit_log_middleware
from shared.utils.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()
app = FastAPI(title="SmartBank AI — Credit Risk Agent", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.middleware("http")(audit_log_middleware)

scorecard_model = None
lgbm_model = None
_start_time = time.time()

SCORE_MIN, SCORE_MAX = 300, 850

# Decision thresholds (configurable per bank)
APPROVE_THRESHOLD = 620
REFER_THRESHOLD = 520


@app.on_event("startup")
async def load_models():
    global scorecard_model, lgbm_model
    model_dir = Path(settings.model_dir)

    sc_path = model_dir / "credit_scorecard.pkl"
    lgbm_path = model_dir / "credit_lgbm.pkl"

    if sc_path.exists():
        with open(sc_path, "rb") as f:
            scorecard_model = pickle.load(f)
        logger.info("Scorecard model loaded")

    if lgbm_path.exists():
        with open(lgbm_path, "rb") as f:
            lgbm_model = pickle.load(f)
        logger.info("LightGBM challenger model loaded")


def compute_score(app: LoanApplicationRequest) -> tuple[float, float, list[dict]]:
    """Returns (credit_score_300_850, pd_probability, top_factors)."""
    dti = app.existing_monthly_obligations_ngn / max(app.monthly_income_ngn, 1)
    ltv = app.loan_amount_ngn / max(app.monthly_income_ngn * 12, 1)

    features = np.array([[
        app.age,
        app.monthly_income_ngn,
        1 if app.employment_type == "salaried" else 0,
        app.loan_amount_ngn,
        app.loan_tenure_months,
        dti,
        ltv,
        app.repayment_history_score,
        1 if app.bvn_verified else 0,
        app.account_age_months,
        app.avg_monthly_balance_ngn,
    ]])

    if lgbm_model is not None:
        pd_prob = float(lgbm_model.predict_proba(features)[0][1])
    else:
        # Stub scoring
        pd_prob = max(0.01, min(0.99, 0.5 - (app.repayment_history_score / 200) + dti * 0.3))

    # Map PD to 300–850 score (inverse relationship)
    credit_score = SCORE_MAX - int((pd_prob ** 0.5) * (SCORE_MAX - SCORE_MIN))
    credit_score = max(SCORE_MIN, min(SCORE_MAX, credit_score))

    # Top factors (simplified — SHAP in production)
    factors = []
    if dti > 0.5:
        factors.append({"factor": "High debt-to-income ratio", "direction": "negative", "value": round(dti, 2)})
    if app.account_age_months < 12:
        factors.append({"factor": "Short account history", "direction": "negative", "value": app.account_age_months})
    if app.repayment_history_score >= 80:
        factors.append({"factor": "Strong repayment history", "direction": "positive", "value": app.repayment_history_score})
    if not app.bvn_verified:
        factors.append({"factor": "BVN not verified", "direction": "negative", "value": False})
    if app.avg_monthly_balance_ngn > app.loan_amount_ngn * 0.1:
        factors.append({"factor": "Adequate average balance", "direction": "positive", "value": round(app.avg_monthly_balance_ngn, 0)})

    return credit_score, pd_prob, factors[:3]


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        agent="credit_risk",
        version="1.0.0",
        model_loaded=lgbm_model is not None or scorecard_model is not None,
        uptime_seconds=round(time.time() - _start_time, 1),
    )


@app.post("/predict", response_model=AgentResponse, dependencies=[Depends(verify_service_token)])
async def predict(loan_app: LoanApplicationRequest):
    start = time.monotonic()
    credit_score, pd_prob, factors = compute_score(loan_app)

    decision = (
        "APPROVE" if credit_score >= APPROVE_THRESHOLD else
        "REFER" if credit_score >= REFER_THRESHOLD else
        "DECLINE"
    )

    narrative = (
        f"Credit score: {credit_score}/850. Probability of default: {round(pd_prob * 100, 1)}%. "
        f"Decision: {decision}. "
        + ("Key positive factors: " if any(f["direction"] == "positive" for f in factors) else "Key concerns: ")
        + "; ".join(f["factor"] for f in factors) + "."
    )

    latency = (time.monotonic() - start) * 1000
    return AgentResponse(
        agent="credit_risk",
        version="1.0.0",
        latency_ms=round(latency, 2),
        payload={
            "customer_id": loan_app.customer_id,
            "credit_score": credit_score,
            "probability_of_default": round(pd_prob, 4),
            "decision": decision,
            "max_recommended_loan_ngn": round(loan_app.monthly_income_ngn * 6, 0) if decision != "DECLINE" else 0,
            "top_factors": factors,
            "narrative": narrative,
        },
    )
