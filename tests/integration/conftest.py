"""Integration harness for the orchestrator-to-agent boundary.

Both rounds of P0 defects were contract mismatches *between* services: the
orchestrator sent a shape the agent did not serve. Unit tests cannot see that,
because each side is individually correct.

This harness wires the real orchestrator app to the real agent apps over real
HTTP semantics — the orchestrator's own `httpx` calls are routed into the agent
ASGI applications instead of the network. Request validation, status codes and
response models are all genuine; only the socket is not. That runs in seconds
with no Docker, so it can gate pull requests, which the compose-based job
cannot (it is skipped on `pull_request`).

Tests marked `live` run against a real compose stack instead and are skipped
unless SMARTBANK_INTEGRATION_BASE_URL is set.
"""

import os
import sys
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

DEFAULT_TOKEN = "integration-service-token-not-a-real-secret"
PLATFORM_CLIENT_ID = "smartbank-platform"

# Settings are read through an lru_cache at import time, so the environment must
# be established before any service module is imported.
os.environ.setdefault("SERVICE_AUTH_TOKEN", DEFAULT_TOKEN)
# Read back: CI and developer shells may already define one, and the tests must
# present whatever the services are actually configured with.
SERVICE_TOKEN = os.environ["SERVICE_AUTH_TOKEN"]
os.environ.setdefault("ORCHESTRATOR_ALLOWED_CLIENT_ID", PLATFORM_CLIENT_ID)
os.environ.setdefault("ENVIRONMENT", "development")
os.environ.setdefault("SMARTBANK_ALLOW_UNVERIFIED_ARTEFACTS", "true")

AGENT_MODULES = {
    "fraud-detection": "agents.fraud_detection.main",
    "credit-risk": "agents.credit_risk.main",
    "aml-compliance": "agents.aml_compliance.main",
    "personalization": "agents.personalization.main",
    "predictive-analytics": "agents.predictive_analytics.main",
    "conversational-ai": "agents.conversational_ai.main",
    "smart-dashboard": "agents.smart_dashboard.main",
    "data-aggregation": "agents.data_aggregation.main",
}


@pytest.fixture(scope="session")
def agent_apps():
    """Import every agent application once per session."""
    import importlib

    from shared.utils.config import get_settings

    get_settings.cache_clear()
    return {host: importlib.import_module(module).app for host, module in AGENT_MODULES.items()}


@pytest.fixture(scope="session")
def orchestrator_module(agent_apps):
    """The orchestrator, with its outbound HTTP client routed to the agent apps."""
    import functools
    import importlib

    import httpx

    from shared.utils.config import get_settings

    get_settings.cache_clear()
    orchestrator = importlib.import_module("orchestrator.main")

    transports = {host: httpx.ASGITransport(app=app) for host, app in agent_apps.items()}

    class AgentRouterTransport(httpx.AsyncBaseTransport):
        """Dispatch by hostname to the matching agent application."""

        def __init__(self):
            self.unreachable: set[str] = set()

        async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
            host = request.url.host
            if host in self.unreachable:
                raise httpx.ConnectError(f"simulated outage for {host}", request=request)
            transport = transports.get(host)
            if transport is None:
                raise httpx.ConnectError(f"no agent registered for {host}", request=request)
            return await transport.handle_async_request(request)

    router = AgentRouterTransport()
    orchestrator.httpx.AsyncClient = functools.partial(httpx.AsyncClient, transport=router)
    orchestrator.router_transport = router
    return orchestrator


@pytest.fixture
def router(orchestrator_module):
    """Control which agents are reachable for a given test."""
    transport = orchestrator_module.router_transport
    transport.unreachable.clear()
    yield transport
    transport.unreachable.clear()


@pytest.fixture
def reset_circuit_breaker(orchestrator_module):
    """The breaker is process-global; clear it so tests do not leak into each other."""
    breaker = orchestrator_module.circuit_breaker
    for agent in orchestrator_module.AGENT_URLS:
        breaker.record_success(agent)
    yield breaker
    for agent in orchestrator_module.AGENT_URLS:
        breaker.record_success(agent)


@pytest.fixture
def client(orchestrator_module, router, reset_circuit_breaker):
    """An httpx client bound to the orchestrator app, with platform credentials."""
    import httpx

    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=orchestrator_module.app),
        base_url="http://orchestrator",
        headers={"X-Service-Token": SERVICE_TOKEN, "X-Client-ID": PLATFORM_CLIENT_ID},
    )


@pytest.fixture
def anonymous_client(orchestrator_module, router):
    """A client with no platform credentials, for negative-authorisation tests."""
    import httpx

    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=orchestrator_module.app),
        base_url="http://orchestrator",
    )
