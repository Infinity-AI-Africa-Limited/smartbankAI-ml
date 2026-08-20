"""
Shared Pydantic schemas for SmartBank AI agent APIs.
All agents import from this module to ensure consistent request/response contracts.
"""
from pydantic import BaseModel, ConfigDict, Field
from typing import Any, Literal, Optional
from datetime import datetime, timezone
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
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
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
    # Not a model input: the scorecard's feature set uses account_age_months, not
    # applicant age. Carried as optional metadata only, with no default, so that
    # nothing downstream can mistake a filler value for a supplied one.
    age: Optional[int] = None
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
    # Mirrors CustomerFeatures in the v1 contract. Everything the platform is
    # allowed to omit under payload minimisation is optional here too, otherwise
    # a correctly minimised request is rejected 422 at the agent.
    customer_id: str
    products_held: list[str]
    channel_preference: str  # mobile | web | ussd | branch
    account_age_months: int
    age_band: Optional[str] = None  # 18-25 | 26-35 | 36-45 | 46-55 | 55+
    income_band: Optional[str] = None  # low | mid | high | premium
    days_since_last_transaction: Optional[int] = None
    monthly_txn_count_3m_avg: Optional[float] = None
    complaint_count_12m: Optional[int] = None


# ── Conversational AI schema ──────────────────────────────────────────────────

class ConversationTurn(BaseModel):
    """One prior turn. Typed so a caller cannot inject arbitrary roles or shapes."""
    model_config = ConfigDict(extra="forbid")

    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=4000)


class ConversationalRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: str = Field(min_length=1, max_length=100)
    customer_id: Optional[str] = Field(default=None, max_length=100)
    message: str = Field(min_length=1, max_length=4000)
    conversation_history: list[ConversationTurn] = Field(default_factory=list, max_length=20)
    language: str = Field(default="en", min_length=2, max_length=10)


class ConversationalResponse(BaseModel):
    session_id: str
    response: str
    sources: list[str] = []
    suggested_actions: list[str] = []
    # Optional: omitted unless a calibrated score exists. A constant written here
    # would land in the advisory audit record as if it had been measured.
    confidence: Optional[float] = None
