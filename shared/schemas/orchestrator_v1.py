"""Versioned private contract between the SmartBank platform backend and ML orchestrator."""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field


CONTRACT_VERSION = "2026-08-01"


class RequestType(str, Enum):
    FRAUD_CHECK = "fraud_check"
    CREDIT_ASSESSMENT = "credit_assessment"
    AML_CHECK = "aml_check"
    RECOMMEND = "recommend"
    CHAT = "chat"


class ContractRequest(BaseModel):
    """Base request metadata that makes every prediction traceable and tenant-scoped."""

    model_config = ConfigDict(extra="forbid")

    contract_version: Literal[CONTRACT_VERSION]
    correlation_id: UUID
    tenant_id: str = Field(min_length=1, max_length=100)
    request_type: RequestType
    requested_at: datetime


class TransactionFeatures(BaseModel):
    """Minimised transaction features; account numbers and PII are intentionally excluded."""

    model_config = ConfigDict(extra="forbid")

    transaction_id: str = Field(min_length=1, max_length=100)
    amount_ngn: float = Field(ge=0)
    channel: Literal["web", "mobile", "ussd", "pos", "atm", "branch", "api"]
    hour_of_day: int = Field(ge=0, le=23)
    day_of_week: int = Field(ge=0, le=6)
    merchant_category: Optional[str] = Field(default=None, max_length=100)
    origin_region: Optional[str] = Field(default=None, max_length=100)
    sender_30d_avg_amount: Optional[float] = Field(default=None, ge=0)
    sender_txn_count_1h: Optional[int] = Field(default=None, ge=0)


class CreditFeatures(BaseModel):
    """Minimised credit features; BVN and NIN values must never be sent to the ML API."""

    model_config = ConfigDict(extra="forbid")

    customer_id: str = Field(min_length=1, max_length=100)
    monthly_income_ngn: float = Field(ge=0)
    employment_type: Literal["salaried", "self_employed", "informal", "unemployed", "unknown"]
    loan_amount_ngn: float = Field(ge=0)
    loan_tenure_months: int = Field(ge=1, le=120)
    existing_monthly_obligations_ngn: float = Field(ge=0)
    repayment_history_score: float = Field(ge=0, le=100)
    bvn_verified: bool
    account_age_months: int = Field(ge=0)
    avg_monthly_balance_ngn: float = Field(ge=0)


class CustomerFeatures(BaseModel):
    model_config = ConfigDict(extra="forbid")

    customer_id: str = Field(min_length=1, max_length=100)
    products_held: list[str] = Field(max_length=30)
    channel_preference: Literal["mobile", "web", "ussd", "branch", "unknown"]
    account_age_months: int = Field(ge=0)
    age_band: Optional[Literal["18-25", "26-35", "36-45", "46-55", "55+", "unknown"]] = None
    income_band: Optional[Literal["low", "mid", "high", "premium", "unknown"]] = None
    days_since_last_transaction: Optional[int] = Field(default=None, ge=0)
    monthly_txn_count_3m_avg: Optional[float] = Field(default=None, ge=0)
    complaint_count_12m: Optional[int] = Field(default=None, ge=0)


class ConversationTurn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=4000)


class AssistantFeatures(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: str = Field(min_length=1, max_length=100)
    message: str = Field(min_length=1, max_length=4000)
    customer_id: Optional[str] = Field(default=None, min_length=1, max_length=100)
    conversation_history: list[ConversationTurn] = Field(default_factory=list, max_length=20)
    language: str = Field(default="en", min_length=2, max_length=10)


class OrchestratorRequestV1(ContractRequest):
    """Validated payload envelope accepted by POST /v1/route."""

    payload: dict[str, Any]


class ModelMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    agent: str
    model_name: Optional[str] = None
    model_version: Optional[str] = None


class AdvisoryExplanation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    summary: Optional[str] = Field(default=None, max_length=4000)
    top_factors: list[dict[str, Any]] = Field(default_factory=list, max_length=20)


class AdvisoryResponseV1(BaseModel):
    """Normalised, non-autonomous result returned to the platform backend."""

    contract_version: Literal[CONTRACT_VERSION] = CONTRACT_VERSION
    correlation_id: UUID
    decision_id: UUID = Field(default_factory=uuid4)
    request_type: RequestType
    status: Literal["advisory", "unavailable", "rejected"]
    recommendation: Optional[str] = Field(default=None, max_length=4000)
    confidence: Optional[float] = Field(default=None, ge=0, le=1)
    human_review_required: Literal[True] = True
    explanation: Optional[AdvisoryExplanation] = None
    model: Optional[ModelMetadata] = None
    received_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    latency_ms: Optional[float] = Field(default=None, ge=0)


class HealthResponseV1(BaseModel):
    status: Literal["ok", "degraded"]
    contract_versions: list[str]
    agents: dict[str, str]
