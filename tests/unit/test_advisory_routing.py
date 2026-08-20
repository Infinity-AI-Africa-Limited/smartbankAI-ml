"""Regression tests for orchestrator advisory normalisation and the AML contract.

Each test pins a defect found during the P0 review:
  * the credit agent's recommendation was dropped because it is published as
    `advisory_recommendation`, not `recommendation`;
  * a multi-agent route produced an empty advisory, and a total agent outage
    still reported `status="advisory"`;
  * `aml_check` was validated against the single-transaction feature model, so
    every AML call 422'd at the agent and tripped the circuit breaker.
"""

import json
from datetime import datetime, timezone
from uuid import uuid4

import pytest
from fastapi import HTTPException

from orchestrator.main import normalise_advisory_result, validate_payload
from shared.schemas.orchestrator_v1 import CONTRACT_VERSION, OrchestratorRequestV1, RequestType


def make_request(request_type: RequestType = RequestType.CREDIT_ASSESSMENT) -> OrchestratorRequestV1:
    return OrchestratorRequestV1(
        contract_version=CONTRACT_VERSION,
        correlation_id=uuid4(),
        tenant_id="tenant-1",
        request_type=request_type,
        requested_at=datetime.now(timezone.utc),
        payload={},
    )


CREDIT_RESULT = {
    "agent": "credit_risk",
    "payload": {
        "customer_id": "CUST-1",
        "credit_score": 712,
        "advisory_recommendation": "REFER_FOR_APPROVAL",
        "top_factors": [{"factor": "repayment history", "direction": "positive"}],
        "narrative": "Credit score 712/850. Human credit-officer review is required.",
        "human_review_required": True,
    },
}

FRAUD_RESULT = {
    "agent": "fraud_detection",
    "payload": {
        "top_factors": [{"factor": "device risk", "direction": "negative"}],
        "narrative": "No elevated transaction risk observed.",
    },
}

UNAVAILABLE = {"agent": "credit_risk", "status": "unavailable", "fallback": True, "message": "down"}


def test_credit_recommendation_is_preserved():
    result = normalise_advisory_result(make_request(), [("credit_risk", CREDIT_RESULT)], 12.0)
    assert result.recommendation == "REFER_FOR_APPROVAL"
    assert result.status == "advisory"
    assert result.human_review_required is True


def test_multi_agent_route_keeps_the_primary_recommendation_and_tags_factors():
    result = normalise_advisory_result(
        make_request(), [("credit_risk", CREDIT_RESULT), ("fraud_detection", FRAUD_RESULT)], 20.0
    )
    assert result.recommendation == "REFER_FOR_APPROVAL"
    assert result.model is not None and result.model.agent == "credit_risk"
    agents = {factor["agent"] for factor in result.explanation.top_factors}
    assert agents == {"credit_risk", "fraud_detection"}


def test_primary_agent_outage_is_reported_unavailable():
    result = normalise_advisory_result(
        make_request(), [("credit_risk", UNAVAILABLE), ("fraud_detection", FRAUD_RESULT)], 8.0
    )
    assert result.status == "unavailable"
    assert result.recommendation is not None and "human-review" in result.recommendation


def test_supporting_agent_outage_stays_advisory_but_is_disclosed():
    degraded = {"agent": "fraud_detection", "status": "error", "fallback": True}
    result = normalise_advisory_result(
        make_request(), [("credit_risk", CREDIT_RESULT), ("fraud_detection", degraded)], 8.0
    )
    assert result.status == "advisory"
    assert result.recommendation == "REFER_FOR_APPROVAL"
    assert "fraud_detection" in result.explanation.summary
    assert "partial" in result.explanation.summary


def test_no_agent_result_is_never_reported_as_advisory():
    result = normalise_advisory_result(make_request(), [], 1.0)
    assert result.status == "unavailable"


AML_PAYLOAD = {
    "customer_id": "CUST-KEY-1",
    "transactions": [
        {
            "id": "TXN-1",
            "sender": "CUST-KEY-1",
            "receiver": "CUST-KEY-2",
            "amount_ngn": 920000,
            "timestamp": "2026-08-20T09:00:00+00:00",
        }
    ],
}


def test_aml_check_accepts_the_shape_the_agent_actually_serves():
    validated = validate_payload(RequestType.AML_CHECK, AML_PAYLOAD)
    assert validated["customer_id"] == "CUST-KEY-1"
    assert validated["check_types"] == ["structuring", "layering", "smurfing"]
    # Must survive the httpx json= encode on the way to the agent.
    assert isinstance(validated["transactions"][0]["timestamp"], str)
    json.dumps(validated)


def test_aml_check_rejects_a_single_transaction_feature_payload():
    with pytest.raises(HTTPException) as excinfo:
        validate_payload(
            RequestType.AML_CHECK,
            {"transaction_id": "T1", "amount_ngn": 1000, "channel": "mobile", "hour_of_day": 9, "day_of_week": 2},
        )
    assert excinfo.value.status_code == 422


def test_aml_party_keys_reject_raw_bvn_values():
    payload = json.loads(json.dumps(AML_PAYLOAD))
    payload["transactions"][0]["sender"] = "22123456789"
    with pytest.raises(HTTPException) as excinfo:
        validate_payload(RequestType.AML_CHECK, payload)
    assert excinfo.value.status_code == 422
