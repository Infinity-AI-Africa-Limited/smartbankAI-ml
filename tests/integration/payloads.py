"""Shared request payloads for the integration suites.

Kept in one module so the in-process contract suite and the live-stack suite
cannot drift apart: if the contract changes, both fail together.
"""

import os
from datetime import datetime, timedelta, timezone
from uuid import uuid4

from shared.schemas.orchestrator_v1 import CONTRACT_VERSION, RequestType

NOW = datetime.now(timezone.utc)

# conftest establishes this before any test module is imported.
SERVICE_TOKEN = os.environ.get("SERVICE_AUTH_TOKEN", "integration-service-token-not-a-real-secret")


def envelope(request_type: RequestType, payload: dict) -> dict:
    return {
        "contract_version": CONTRACT_VERSION,
        "correlation_id": str(uuid4()),
        "tenant_id": "bank-tenant-1",
        "request_type": request_type.value,
        "requested_at": NOW.isoformat(),
        "payload": payload,
    }


# Minimised payloads - exactly what a payload-minimising platform gateway sends.
TRANSACTION = {
    "transaction_id": "TXN-2026-0001",
    "amount_ngn": 480000.0,
    "channel": "mobile",
    "hour_of_day": 22,
    "day_of_week": 5,
}

CREDIT = {
    "customer_id": "CUST-KEY-9001",
    "monthly_income_ngn": 850000.0,
    "employment_type": "salaried",
    "loan_amount_ngn": 3000000.0,
    "loan_tenure_months": 24,
    "existing_monthly_obligations_ngn": 120000.0,
    "repayment_history_score": 78.0,
    "bvn_verified": True,
    "account_age_months": 36,
    "avg_monthly_balance_ngn": 410000.0,
}

AML = {
    "customer_id": "CUST-KEY-9001",
    "transactions": [
        {
            "id": "TXN-{0}".format(index),
            "sender": "CUST-KEY-9001",
            "receiver": "CUST-KEY-{0}".format(9100 + index),
            "amount_ngn": 950000.0,
            "timestamp": (NOW - timedelta(hours=index + 1)).isoformat(),
        }
        for index in range(4)
    ],
}

RECOMMEND = {
    "customer_id": "CUST-KEY-9001",
    "products_held": ["savings_account"],
    "channel_preference": "mobile",
    "account_age_months": 36,
}

CHAT = {"session_id": "SESSION-1", "message": "What is my account balance?"}

ROUND_TRIPS = [
    (RequestType.FRAUD_CHECK, TRANSACTION),
    (RequestType.CREDIT_ASSESSMENT, CREDIT),
    (RequestType.AML_CHECK, AML),
    (RequestType.RECOMMEND, RECOMMEND),
    (RequestType.CHAT, CHAT),
]


