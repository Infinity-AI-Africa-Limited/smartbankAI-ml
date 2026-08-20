"""End-to-end checks against a running stack.

Skipped unless SMARTBANK_INTEGRATION_BASE_URL points at a live orchestrator, so
the default test run needs no Docker. Bring the stack up with:

    SERVICE_AUTH_TOKEN=<32+ chars> docker compose -f infra/docker/docker-compose.yml up -d
    SMARTBANK_INTEGRATION_BASE_URL=http://localhost:8001 pytest tests/integration -m live

These cover what the in-process contract suite cannot: real sockets, real
container networking, real startup behaviour, and the negative-connectivity
evidence the private-staging gate requires.
"""

import os

import httpx
import pytest

from payloads import AML, CHAT, CREDIT, RECOMMEND, ROUND_TRIPS, TRANSACTION, envelope
from shared.schemas.orchestrator_v1 import RequestType

BASE_URL = os.environ.get("SMARTBANK_INTEGRATION_BASE_URL")
TOKEN = os.environ.get("SERVICE_AUTH_TOKEN", "")
CLIENT_ID = os.environ.get("ORCHESTRATOR_ALLOWED_CLIENT_ID", "smartbank-platform")

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(not BASE_URL, reason="SMARTBANK_INTEGRATION_BASE_URL is not set"),
]

# Agent ports must not be published to the host: the platform gateway is the
# only permitted caller and reaches them over the private compose network.
PRIVATE_AGENT_PORTS = [8002, 8003, 8004, 8005, 8006, 8007, 8008, 8009]

PAYLOADS = {
    RequestType.FRAUD_CHECK: TRANSACTION,
    RequestType.CREDIT_ASSESSMENT: CREDIT,
    RequestType.AML_CHECK: AML,
    RequestType.RECOMMEND: RECOMMEND,
    RequestType.CHAT: CHAT,
}


def platform_client() -> httpx.Client:
    return httpx.Client(
        base_url=BASE_URL,
        timeout=30.0,
        headers={"X-Service-Token": TOKEN, "X-Client-ID": CLIENT_ID},
    )


def test_orchestrator_reports_every_agent_healthy():
    with httpx.Client(base_url=BASE_URL, timeout=30.0) as client:
        response = client.get("/health")

    assert response.status_code == 200, response.text
    body = response.json()
    unhealthy = {name: state for name, state in body["agents"].items() if state != "ok"}
    assert not unhealthy, "agents not healthy: {0}".format(unhealthy)


@pytest.mark.parametrize("request_type", list(PAYLOADS), ids=lambda rt: rt.value)
def test_every_request_type_round_trips_over_the_network(request_type):
    with platform_client() as client:
        response = client.post("/v1/route", json=envelope(request_type, PAYLOADS[request_type]))

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] == "advisory", body
    assert body["human_review_required"] is True


def test_unauthenticated_caller_is_rejected():
    with httpx.Client(base_url=BASE_URL, timeout=30.0) as client:
        response = client.post(
            "/v1/route", json=envelope(RequestType.FRAUD_CHECK, TRANSACTION)
        )
    assert response.status_code in (401, 403, 503)


@pytest.mark.parametrize("port", PRIVATE_AGENT_PORTS)
def test_agents_are_not_reachable_from_the_host(port):
    """Negative-connectivity evidence for the private-staging gate."""
    host = httpx.URL(BASE_URL).host
    try:
        response = httpx.get("http://{0}:{1}/health".format(host, port), timeout=5.0)
    except httpx.HTTPError:
        return  # refused or timed out: the agent is private, which is what we want
    pytest.fail(
        "agent on port {0} answered from the host with {1}; only the orchestrator "
        "may be published".format(port, response.status_code)
    )


@pytest.mark.parametrize("request_type", list(PAYLOADS), ids=lambda rt: rt.value)
def test_advisory_responses_carry_model_provenance(request_type):
    """Evidence that a model actually loaded and served the request.

    The orchestrator's /health reports reachability only, so it cannot show
    model_loaded. An advisory that comes back naming the agent that produced it
    is the evidence available under the current health contract. Surfacing
    model_loaded per agent needs a health-contract revision, re-pinned in both
    repositories.
    """
    with platform_client() as client:
        response = client.post("/v1/route", json=envelope(request_type, PAYLOADS[request_type]))

    body = response.json()
    assert body["status"] == "advisory", body
    assert body["model"] is not None, f"no model provenance recorded: {body}"
    assert body["model"]["agent"], body["model"]


def test_the_removed_legacy_route_is_not_served():
    with platform_client() as client:
        response = client.post("/route", json={"request_type": "fraud_check", "payload": {}})
    assert response.status_code == 404


def test_round_trip_payload_set_matches_the_contract_suite():
    """Guard against the two suites drifting apart."""
    assert {rt for rt, _ in ROUND_TRIPS} == set(PAYLOADS)
