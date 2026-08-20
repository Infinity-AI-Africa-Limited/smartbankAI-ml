"""
Shared Pydantic schemas for SmartBank AI agent APIs.
All agents import from this module to ensure consistent request/response contracts.
"""
from pydantic import BaseModel, Field
from typing import Any, Optional
from datetime import datetime
from enum import Enum


class RiskLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class AgentResponse(BaseModel):
    """Base response envelope returned by every agent endpoint."""
    agent: str
    version: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    latency_ms: Optional[float] = None
    payload: Any


class HealthResponse(BaseModel):
    status: str = "ok"
    agent: str
    version: str
    model_loaded: bool
    uptime_seconds: float


class ExplainResponse(BaseModel):
    score: float
    risk_level: RiskLevel
    top_factors: list[dict]
    narrative: str


# ── Transaction schema (used by Fraud + AML agents) ──────────────────────────

class TransactionRequest(BaseModel):
    transaction_id: str
    amount_ngn: float
    # The v1 platform contract sends no full account identifiers. These legacy fields
    # remain optional to support older internal callers while models use derived features.
    sender_account: Optional[str] = None
    receiver_account: Optional[str] = None
    channel: str  # mobile | web | ussd | atm | pos | branch | api
    merchant_category: Optional[str] = None
    hour_of_day: int
    day_of_week: int
    device_id: Optional[str] = None
    location: Optional[str] = None
    sender_30d_avg_amount: Optional[float] = None
    sender_txn_count_1h: Optional[int] = None


# ── Loan application schema (used by Credit Risk agent) ───────────────────────

class LoanApplicationRequest(BaseModel):
    customer_id: str
    # Age is intentionally optional in the v1 minimised contract. The current
    # scorecard uses a neutral fallback until a reviewed replacement is trained.
    age: int = 35
    monthly_income_ngn: float
    employment_type: str  # salaried | self_employed | informal | unemployed
    loan_amount_ngn: float
    loan_tenure_months: int
    existing_monthly_obligations_ngn: float
    repayment_history_score: float  # 0–100
    bvn_verified: bool
    account_age_months: int
    avg_monthly_balance_ngn: float


# ── Customer profile schema (used by Personalization + Predictive agents) ─────

class CustomerProfileRequest(BaseModel):
    customer_id: str
    age_band: str  # 18-25 | 26-35 | 36-45 | 46-55 | 55+
    income_band: str  # low | mid | high | premium
    products_held: list[str]
    channel_preference: str  # mobile | web | ussd | branch
    days_since_last_transaction: int
    monthly_txn_count_3m_avg: float
    complaint_count_12m: int
    account_age_months: int


# ── Conversational AI schema ──────────────────────────────────────────────────

class ConversationalRequest(BaseModel):
    session_id: str
    customer_id: Optional[str] = None
    message: str
    conversation_history: list[dict] = []
    language: str = "en"


class ConversationalResponse(BaseModel):
    session_id: str
    response: str
    sources: list[str] = []
    suggested_actions: list[str] = []
    confidence: float
