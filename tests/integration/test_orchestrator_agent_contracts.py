"""Contract tests across the orchestrator-to-agent boundary.

Every defect found in the two P0 review rounds lived here: each service was
internally correct while disagreeing with its neighbour about the payload.
"""

import pytest

from payloads import (
    AML,
    CREDIT,
    ROUND_TRIPS,
    SERVICE_TOKEN,
    TRANSACTION,
    envelope,
)
from shared.schemas.orchestrator_v1 import CONTRACT_VERSION, RequestType


# -- Structural ---------------------------------------------------------------

def test_every_route_targets_an_endpoint_the_agent_actually_serves(orchestrator_module, agent_apps):
    """The check that would have caught the AML mismatch immediately."""
    host_by_agent = {
        agent: orchestrator_module.AGENT_URLS[agent].split("//")[1].split(":")[0]
        for agent in orchestrator_module.AGENT_URLS
    }
    missing = []
    for request_type, routes in orchestrator_module.ROUTE_MAP.items():
        for agent, endpoint in routes:
            app = agent_apps[host_by_agent[agent]]
            served = {route.path for route in app.routes}
            if endpoint not in served:
                missing.append("{0} -> {1}{2} (serves {3})".format(request_type, agent, endpoint, sorted(served)))
    assert not missing, "routes point at endpoints no agent serves:\n" + "\n".join(missing)


# -- Round trips --------------------------------------------------------------

@pytest.mark.parametrize(("request_type", "payload"), ROUND_TRIPS, ids=[rt.value for rt, _ in ROUND_TRIPS])
async def test_request_round_trips_to_an_advisory_response(client, request_type, payload):
    async with client:
        response = await client.post("/v1/route", json=envelope(request_type, payload))

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] == "advisory", "agent rejected the minimised payload: {0}".format(body)
    assert body["human_review_required"] is True
    assert body["contract_version"] == CONTRACT_VERSION


async def test_credit_assessment_returns_a_referral_recommendation(client):
    """Regression: the credit agent's advisory_recommendation was being dropped."""
    async with client:
        response = await client.post("/v1/route", json=envelope(RequestType.CREDIT_ASSESSMENT, CREDIT))

    body = response.json()
    assert body["recommendation"] is not None, "credit recommendation was dropped in normalisation"
    assert body["recommendation"].startswith("REFER"), body["recommendation"]
    assert body["explanation"]["top_factors"], "credit factors did not survive normalisation"


async def test_aml_check_reaches_the_agent_rather_than_failing_validation(client):
    """Regression: aml_check was validated against the single-transaction model."""
    async with client:
        response = await client.post("/v1/route", json=envelope(RequestType.AML_CHECK, AML))

    body = response.json()
    assert body["status"] == "advisory", "AML payload was rejected by the agent: {0}".format(body)
    assert body["model"]["agent"] == "aml_compliance"


@pytest.mark.parametrize(("request_type", "payload"), ROUND_TRIPS, ids=[rt.value for rt, _ in ROUND_TRIPS])
async def test_confidence_is_absent_or_calibrated(client, request_type, payload):
    """No agent may publish an invented confidence into the audit record."""
    async with client:
        response = await client.post("/v1/route", json=envelope(request_type, payload))

    confidence = response.json().get("confidence")
    assert confidence is None or 0.0 <= confidence <= 1.0


# -- Degradation --------------------------------------------------------------

async def test_primary_agent_outage_is_reported_unavailable(client, router):
    router.unreachable.add("credit-risk")
    async with client:
        response = await client.post("/v1/route", json=envelope(RequestType.CREDIT_ASSESSMENT, CREDIT))

    body = response.json()
    assert body["status"] == "unavailable", "a failed primary agent must not read as a successful advisory"
    assert body["human_review_required"] is True


async def test_supporting_agent_outage_stays_advisory_and_is_disclosed(client, router):
    router.unreachable.add("fraud-detection")
    async with client:
        response = await client.post("/v1/route", json=envelope(RequestType.CREDIT_ASSESSMENT, CREDIT))

    body = response.json()
    assert body["status"] == "advisory"
    assert body["recommendation"].startswith("REFER")
    assert "fraud_detection" in body["explanation"]["summary"]


async def test_agent_errors_do_not_leak_internals_to_the_platform(client, router):
    router.unreachable.add("fraud-detection")
    async with client:
        response = await client.post("/v1/route", json=envelope(RequestType.FRAUD_CHECK, TRANSACTION))

    serialised = repr(response.json())
    for leak in ("Traceback", "ConnectError", "httpx", "/app/"):
        assert leak not in serialised, "internal detail {0!r} reached the platform".format(leak)


# -- Negative authorisation ---------------------------------------------------

async def test_unauthenticated_request_is_rejected(anonymous_client):
    async with anonymous_client:
        response = await anonymous_client.post("/v1/route", json=envelope(RequestType.FRAUD_CHECK, TRANSACTION))
    assert response.status_code in (401, 403, 503)


async def test_wrong_client_id_is_rejected(orchestrator_module, router):
    import httpx

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=orchestrator_module.app),
        base_url="http://orchestrator",
        headers={"X-Service-Token": SERVICE_TOKEN, "X-Client-ID": "not-the-platform"},
    ) as rogue:
        response = await rogue.post("/v1/route", json=envelope(RequestType.FRAUD_CHECK, TRANSACTION))
    assert response.status_code == 403


async def test_circuit_breaker_controls_require_authentication(anonymous_client):
    async with anonymous_client:
        status = await anonymous_client.get("/circuit-breaker/status")
        reset = await anonymous_client.post("/circuit-breaker/reset/fraud_detection")
    assert status.status_code in (401, 403, 503)
    assert reset.status_code in (401, 403, 503)


# -- Contract enforcement -----------------------------------------------------

async def test_stale_contract_version_is_refused(client):
    body = envelope(RequestType.FRAUD_CHECK, TRANSACTION)
    body["contract_version"] = "2020-01-01"
    async with client:
        response = await client.post("/v1/route", json=body)
    assert response.status_code == 422


async def test_unminimised_payload_is_refused(client):
    """Account numbers must not survive validation even if a caller sends them."""
    body = envelope(RequestType.FRAUD_CHECK, dict(TRANSACTION, sender_account="0123456789"))
    async with client:
        response = await client.post("/v1/route", json=body)
    assert response.status_code == 422


async def test_raw_bvn_in_an_aml_party_field_is_refused(client):
    tampered = dict(AML["transactions"][0], sender="22123456789")
    payload = dict(AML, transactions=[tampered])
    async with client:
        response = await client.post("/v1/route", json=envelope(RequestType.AML_CHECK, payload))
    assert response.status_code == 422


async def test_the_removed_legacy_route_stays_removed(client):
    async with client:
        response = await client.post("/route", json={"request_type": "fraud_check", "payload": {}})
    assert response.status_code == 404
