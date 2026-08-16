from datetime import datetime, timezone
from uuid import uuid4

import pytest
from pydantic import ValidationError

from shared.schemas.orchestrator_v1 import (
    CONTRACT_VERSION,
    AdvisoryResponseV1,
    OrchestratorRequestV1,
    RequestType,
    TransactionFeatures,
)


def test_minimised_transaction_contract_excludes_full_account_numbers():
    payload = TransactionFeatures(
        transaction_id="TXN-001",
        amount_ngn=125000,
        channel="mobile",
        hour_of_day=10,
        day_of_week=2,
    )

    serialised = payload.model_dump()
    assert serialised["transaction_id"] == "TXN-001"
    assert "sender_account" not in serialised
    assert "receiver_account" not in serialised


def test_request_requires_the_pinned_contract_version():
    with pytest.raises(ValidationError):
        OrchestratorRequestV1(
            contract_version="2099-01-01",
            correlation_id=uuid4(),
            tenant_id="4",
            request_type=RequestType.FRAUD_CHECK,
            requested_at=datetime.now(timezone.utc),
            payload={
                "transaction_id": "TXN-001",
                "amount_ngn": 125000,
                "channel": "mobile",
                "hour_of_day": 10,
                "day_of_week": 2,
            },
        )


def test_advisory_response_mandates_human_review():
    response = AdvisoryResponseV1(
        correlation_id=uuid4(),
        request_type=RequestType.CREDIT_ASSESSMENT,
        status="advisory",
        recommendation="REFER",
        human_review_required=True,
    )

    assert response.contract_version == CONTRACT_VERSION
    assert response.human_review_required is True


def test_advisory_response_rejects_autonomous_human_review_flag():
    with pytest.raises(ValidationError):
        AdvisoryResponseV1(
            correlation_id=uuid4(),
            request_type=RequestType.CREDIT_ASSESSMENT,
            status="advisory",
            human_review_required=False,
        )
