"""
SmartBank AI — Agent Orchestrator
Central router with circuit breaker, audit logging, and multi-agent aggregation.
Port: 8001
"""
import time
import logging
import asyncio
from enum import Enum
from collections import defaultdict
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import httpx
import sys
sys.path.append("/app")

from shared.middleware.auth import audit_log_middleware
from shared.utils.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()
app = FastAPI(title="SmartBank AI — Agent Orchestrator", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
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

    return HealthSummary(
        agents=agent_statuses,
        circuit_breaker=circuit_breaker.status(),
        uptime_seconds=round(time.time() - _start_time, 1),
    )


@app.post("/route")
async def route(req: OrchestratorRequest):
    """Main routing endpoint called by the SmartBank AI tRPC server."""
    start = time.monotonic()
    routes = req.require_agents or ROUTE_MAP.get(req.request_type)

    if not routes:
        raise HTTPException(status_code=400, detail=f"Unknown request_type: {req.request_type}")

    # Execute all required agent calls (parallel where multiple agents needed)
    tasks = [call_agent(agent, endpoint, req.payload) for agent, endpoint in routes]
    results = await asyncio.gather(*tasks)

    # Aggregate multi-agent responses
    if len(results) == 1:
        aggregated = results[0]
    else:
        aggregated = {
            "request_type": req.request_type,
            "agents": {routes[i][0]: results[i] for i in range(len(results))},
        }

    latency = (time.monotonic() - start) * 1000
    return {
        "request_type": req.request_type,
        "latency_ms": round(latency, 2),
        "result": aggregated,
    }


@app.get("/circuit-breaker/status")
async def cb_status():
    return circuit_breaker.status()


@app.post("/circuit-breaker/reset/{agent_name}")
async def cb_reset(agent_name: str):
    if agent_name not in AGENT_URLS:
        raise HTTPException(status_code=404, detail=f"Unknown agent: {agent_name}")
    circuit_breaker.record_success(agent_name)
    return {"agent": agent_name, "state": "CLOSED"}
