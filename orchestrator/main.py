"""
SmartBank AI — Agent Orchestrator
Central router with circuit breaker, audit logging, and multi-agent aggregation.
Port: 8001
"""
import time
import logging
import asyncio
from datetime import datetime, timezone
from enum import Enum
from collections import defaultdict
from fastapi import FastAPI, Depends, HTTPException, Request
from pydantic import BaseModel
from typing import Any, Optional
import httpx
import sys
sys.path.append("/app")

from shared.middleware.auth import verify_service_token, audit_log_middleware
from shared.schemas.orchestrator_v1 import (
    AdvisoryExplanation,
    AdvisoryResponseV1,
    CONTRACT_VERSION,
    CreditFeatures,
    CustomerFeatures,
    HealthResponseV1,
    ModelMetadata,
    OrchestratorRequestV1,
    RequestType,
    TransactionFeatures,
    AssistantFeatures,
)
from shared.utils.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()
app = FastAPI(title="SmartBank AI — Agent Orchestrator", version="1.0.0")
app.middleware("http")(audit_log_middleware)

_start_time = time.time()

# ── Agent registry ────────────────────────────────────────────────────────────

AGENT_URLS = {
    "fraud_detection":       "http://fraud-detection:8002",
    "credit_risk":           "http://credit-risk:8003",
    "aml_compliance":        "http://aml-compliance:8004",
    "personalization":       "http://personalization:8005",
    "predictive_analytics":  "http://predictive-analytics:8006",
    "conversational_ai":     "http://conversational-ai:8007",
    "smart_dashboard":       "http://smart-dashboard:8008",
    "data_aggregation":      "http://data-aggregation:8009",
}

# ── Circuit breaker ───────────────────────────────────────────────────────────

class CircuitState(str, Enum):
    CLOSED = "CLOSED"       # Normal operation
    OPEN = "OPEN"           # Failing — reject requests
    HALF_OPEN = "HALF_OPEN" # Testing recovery

class CircuitBreaker:
    FAILURE_THRESHOLD = 5
    RECOVERY_TIMEOUT = 30  # seconds

    def __init__(self):
        self._state: dict[str, CircuitState] = defaultdict(lambda: CircuitState.CLOSED)
        self._failures: dict[str, int] = defaultdict(int)
        self._last_failure: dict[str, float] = defaultdict(float)

    def is_open(self, agent: str) -> bool:
        if self._state[agent] == CircuitState.OPEN:
            if time.time() - self._last_failure[agent] > self.RECOVERY_TIMEOUT:
                self._state[agent] = CircuitState.HALF_OPEN
                return False
            return True
        return False

    def record_success(self, agent: str):
        self._failures[agent] = 0
        self._state[agent] = CircuitState.CLOSED

    def record_failure(self, agent: str):
        self._failures[agent] += 1
        self._last_failure[agent] = time.time()
        if self._failures[agent] >= self.FAILURE_THRESHOLD:
            self._state[agent] = CircuitState.OPEN
            logger.warning("Circuit OPEN for agent: %s", agent)

    def status(self) -> dict:
        return {agent: self._state[agent] for agent in AGENT_URLS}


circuit_breaker = CircuitBreaker()


async def verify_platform_request(request: Request) -> None:
    """Protect the private platform boundary; browser clients must never call this service."""
    await verify_service_token(request)
    if request.headers.get("X-Client-ID") != settings.orchestrator_allowed_client_id:
        raise HTTPException(status_code=403, detail="Unsupported service client")


# ── HTTP client ───────────────────────────────────────────────────────────────

async def call_agent(agent_name: str, endpoint: str, payload: dict, timeout: float = 10.0) -> dict:
    if circuit_breaker.is_open(agent_name):
        logger.warning("Circuit open for %s — returning fallback", agent_name)
        return {
            "agent": agent_name,
            "status": "unavailable",
            "fallback": True,
            "message": f"{agent_name} is temporarily unavailable. Human review required.",
        }

    base_url = AGENT_URLS[agent_name]
    headers = {"X-Service-Token": settings.service_auth_token, "X-Client-ID": "orchestrator"}

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(f"{base_url}{endpoint}", json=payload, headers=headers)
            response.raise_for_status()
            circuit_breaker.record_success(agent_name)
            return response.json()
    except Exception as e:
        circuit_breaker.record_failure(agent_name)
        logger.error("Agent %s call failed: %s", agent_name, e)
        return {
            "agent": agent_name,
            "status": "error",
            "fallback": True,
            "message": f"Agent error: {str(e)[:100]}",
        }


# ── Request models ────────────────────────────────────────────────────────────

class OrchestratorRequest(BaseModel):
    request_type: str  # "fraud_check" | "credit_assessment" | "aml_check" | "recommend" | "chat" | "insights" | "churn" | "normalise"
    payload: dict
    require_agents: Optional[list[str]] = None  # override default routing


class HealthSummary(BaseModel):
    orchestrator: str = "ok"
    agents: dict
    circuit_breaker: dict
    uptime_seconds: float


def validate_payload(request_type: RequestType, payload: dict[str, Any]) -> dict[str, Any]:
    """Validate request-type payloads and reject unnecessary or unexpected fields."""
    model_by_type = {
        RequestType.FRAUD_CHECK: TransactionFeatures,
        RequestType.AML_CHECK: TransactionFeatures,
        RequestType.CREDIT_ASSESSMENT: CreditFeatures,
        RequestType.RECOMMEND: CustomerFeatures,
        RequestType.CHAT: AssistantFeatures,
    }
    try:
        return model_by_type[request_type].model_validate(payload).model_dump(exclude_none=True)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Invalid minimised feature payload: {str(exc)}") from exc


def normalize_confidence(value: Any) -> Optional[float]:
    if not isinstance(value, (int, float)):
        return None
    # Some established agents return confidence as a 0–100 percentage.
    normalized = value / 100 if value > 1 else value
    return max(0.0, min(float(normalized), 1.0))


def normalise_advisory_result(
    request: OrchestratorRequestV1,
    raw_result: dict[str, Any],
    latency_ms: float,
) -> AdvisoryResponseV1:
    """Convert heterogeneous agent outputs into one explicit, human-review-only envelope."""
    is_unavailable = bool(raw_result.get("fallback")) or raw_result.get("status") in {"error", "unavailable"}
    if is_unavailable:
        return AdvisoryResponseV1(
            correlation_id=request.correlation_id,
            request_type=request.request_type,
            status="unavailable",
            recommendation="ML service unavailable; route to the configured human-review workflow.",
            human_review_required=True,
            received_at=datetime.now(timezone.utc),
            latency_ms=round(latency_ms, 2),
        )

    payload = raw_result.get("payload", raw_result)
    if not isinstance(payload, dict):
        payload = {"value": payload}

    recommendation = payload.get("recommendation") or payload.get("decision") or raw_result.get("recommendation")
    if recommendation is not None:
        recommendation = str(recommendation)

    narrative = payload.get("narrative") or payload.get("explanation") or raw_result.get("message")
    factors = payload.get("top_factors") or payload.get("factors") or []
    top_factors = [factor for factor in factors if isinstance(factor, dict)][:20] if isinstance(factors, list) else []

    agent_name = raw_result.get("agent") or payload.get("agent")
    model = ModelMetadata(
        agent=str(agent_name),
        model_name=str(payload["model_name"]) if payload.get("model_name") else None,
        model_version=str(payload["model_version"]) if payload.get("model_version") else None,
    ) if agent_name else None

    return AdvisoryResponseV1(
        correlation_id=request.correlation_id,
        request_type=request.request_type,
        status="advisory",
        recommendation=recommendation,
        confidence=normalize_confidence(payload.get("confidence") or raw_result.get("confidence")),
        human_review_required=True,
        explanation=AdvisoryExplanation(summary=str(narrative) if narrative else None, top_factors=top_factors),
        model=model,
        received_at=datetime.now(timezone.utc),
        latency_ms=round(latency_ms, 2),
    )


# ── Routing logic ─────────────────────────────────────────────────────────────

ROUTE_MAP = {
    "fraud_check":        [("fraud_detection",      "/predict")],
    "fraud_explain":      [("fraud_detection",      "/explain")],
    "credit_assessment":  [("credit_risk",           "/predict"),
                           ("fraud_detection",       "/predict")],   # parallel
    "aml_check":          [("aml_compliance",        "/analyse")],
    "recommend":          [("personalization",       "/recommend")],
    "chat":               [("conversational_ai",     "/chat")],
    "insights":           [("smart_dashboard",       "/insights")],
    "segment":            [("smart_dashboard",       "/segment")],
    "churn":              [("predictive_analytics",  "/predict/churn")],
    "cashflow":           [("predictive_analytics",  "/predict/cashflow")],
    "volume_forecast":    [("predictive_analytics",  "/predict/volume")],
    "normalise":          [("data_aggregation",      "/normalise")],
}


@app.get("/health")
async def health():
    # Check all agent health endpoints in parallel
    async def check_agent(name: str, url: str):
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                r = await client.get(f"{url}/health")
                return name, "ok" if r.status_code == 200 else "degraded"
        except Exception:
            return name, "unreachable"

    results = await asyncio.gather(*[check_agent(n, u) for n, u in AGENT_URLS.items()])
    agent_statuses = dict(results)

    status = "ok" if all(state == "ok" for state in agent_statuses.values()) else "degraded"
    return {
        **HealthResponseV1(status=status, contract_versions=[CONTRACT_VERSION], agents=agent_statuses).model_dump(),
        "circuit_breaker": circuit_breaker.status(),
        "uptime_seconds": round(time.time() - _start_time, 1),
    }


async def route_agents(request_type: str, payload: dict[str, Any], require_agents: Optional[list[str]] = None) -> tuple[dict[str, Any], float]:
    """Route validated data to agents and return aggregate result plus latency."""
    start = time.monotonic()
    routes = require_agents or ROUTE_MAP.get(request_type)

    if not routes:
        raise HTTPException(status_code=400, detail=f"Unknown request_type: {request_type}")

    # Execute all required agent calls (parallel where multiple agents needed)
    tasks = [call_agent(agent, endpoint, payload) for agent, endpoint in routes]
    results = await asyncio.gather(*tasks)

    # Aggregate multi-agent responses
    if len(results) == 1:
        aggregated = results[0]
    else:
        aggregated = {
            "request_type": request_type,
            "agents": {routes[i][0]: results[i] for i in range(len(results))},
        }

    latency = (time.monotonic() - start) * 1000
    return aggregated, latency


@app.post("/v1/route", response_model=AdvisoryResponseV1, dependencies=[Depends(verify_platform_request)])
async def route_v1(req: OrchestratorRequestV1):
    """Private contract endpoint. Every response is advisory and requires human review."""
    payload = validate_payload(req.request_type, req.payload)
    raw_result, latency = await route_agents(req.request_type.value, payload)
    return normalise_advisory_result(req, raw_result, latency)


@app.post("/route", deprecated=True, dependencies=[Depends(verify_platform_request)])
async def route_legacy(req: OrchestratorRequest):
    """Temporary compatibility endpoint; new platform integrations must use POST /v1/route."""
    aggregated, latency = await route_agents(req.request_type, req.payload, req.require_agents)
    return {"request_type": req.request_type, "latency_ms": round(latency, 2), "result": aggregated}


@app.get("/circuit-breaker/status", dependencies=[Depends(verify_platform_request)])
async def cb_status():
    return circuit_breaker.status()


@app.post("/circuit-breaker/reset/{agent_name}", dependencies=[Depends(verify_platform_request)])
async def cb_reset(agent_name: str):
    if agent_name not in AGENT_URLS:
        raise HTTPException(status_code=404, detail=f"Unknown agent: {agent_name}")
    circuit_breaker.record_success(agent_name)
    return {"agent": agent_name, "state": "CLOSED"}
