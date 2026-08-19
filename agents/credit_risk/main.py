"""
Agent 2: Credit Risk Agent
Models: WoE Logistic Regression Scorecard (CBN-explainable) + LightGBM challenger
Port: 8003
"""
import os
import time
import logging
import pandas as pd
from pathlib import Path
from fastapi import FastAPI, Depends
import sys
sys.path.append("/app")

from shared.schemas.base import LoanApplicationRequest, AgentResponse, HealthResponse
from shared.middleware.auth import (
    audit_log_middleware,
    require_secure_configuration,
    verify_service_token,
)
from shared.utils.artefacts import load_verified_artefact
from shared.utils.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()
app = FastAPI(title="SmartBank AI — Credit Risk Agent", version="1.0.0")
app.middleware("http")(audit_log_middleware)

scorecard_model = None
lgbm_model = None
_start_time = time.time()


@app.on_event("startup")
async def enforce_secure_configuration() -> None:
    """Refuse to serve traffic without a usable service token."""
    require_secure_configuration()

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
        scorecard_model = load_verified_artefact(sc_path)
        logger.info("Scorecard model loaded")

    if lgbm_path.exists():
        lgbm_model = load_verified_artefact(lgbm_path)
        logger.info("LightGBM challenger model loaded")


def compute_score(app: LoanApplicationRequest) -> tuple[float, float, list[dict]]:
    """Returns (credit_score_300_850, pd_probability, top_factors)."""
    dti = app.existing_monthly_obligations_ngn / max(app.monthly_income_ngn, 1)
    features = pd.DataFrame([{
        "monthly_income_ngn": app.monthly_income_ngn,
        "loan_amount_requested_ngn": app.loan_amount_ngn,
        "loan_tenure_months": app.loan_tenure_months,
        "existing_monthly_obligations_ngn": app.existing_monthly_obligations_ngn,
        "repayment_history_score": app.repayment_history_score,
        "account_age_months": app.account_age_months,
        "avg_monthly_balance_ngn": app.avg_monthly_balance_ngn,
        "employment_type": app.employment_type,
        "bvn_verified": int(app.bvn_verified),
    }])

    if scorecard_model is not None:
        pd_prob = float(scorecard_model.probability_of_default(features)[0])
    elif lgbm_model is not None:
        challenger_features = pd.get_dummies(features, columns=["employment_type"], dtype=float)
        columns = lgbm_model.get("columns", challenger_features.columns.tolist()) if isinstance(lgbm_model, dict) else challenger_features.columns.tolist()
        challenger_features = challenger_features.reindex(columns=columns, fill_value=0)
        challenger = lgbm_model["model"] if isinstance(lgbm_model, dict) else lgbm_model
        pd_prob = float(challenger.predict_proba(challenger_features)[0][1])
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


# Affordability ceiling. This is bank credit policy, not a model output: it is
# named and versioned here so a credit-risk owner can approve or change it
# without a code review of the serving path. Override per environment.
AFFORDABILITY_INCOME_MULTIPLE = float(os.getenv("SMARTBANK_AFFORDABILITY_INCOME_MULTIPLE", "6"))
AFFORDABILITY_POLICY_ID = os.getenv("SMARTBANK_AFFORDABILITY_POLICY_ID", "unapproved-development-default")


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

    recommendation = (
        "REFER_FOR_APPROVAL" if credit_score >= APPROVE_THRESHOLD else
        "REFER_FOR_REVIEW" if credit_score >= REFER_THRESHOLD else
        "REFER_FOR_ALTERNATIVE_OPTIONS"
    )

    narrative = (
        f"Credit score: {credit_score}/850. Probability of default: {round(pd_prob * 100, 1)}%. "
        f"Advisory recommendation: {recommendation}. Human credit-officer review is required. "
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
            "advisory_recommendation": recommendation,
            "max_recommended_loan_ngn": round(loan_app.monthly_income_ngn * AFFORDABILITY_INCOME_MULTIPLE, 0) if recommendation != "REFER_FOR_ALTERNATIVE_OPTIONS" else 0,
            "affordability_policy": AFFORDABILITY_POLICY_ID,
            "top_factors": factors,
            "narrative": narrative,
            "human_review_required": True,
            "synthetic_model": scorecard_model is not None or lgbm_model is not None,
        },
    )
