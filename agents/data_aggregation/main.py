"""
Agent 8: Data Aggregation Agent
Components: Multi-format ETL normaliser + BVN-based entity resolution
Port: 8009
"""
import time
import logging
import hashlib
import xml.etree.ElementTree as ET
from fastapi import FastAPI, Depends
from pydantic import BaseModel
from typing import Optional
import sys
sys.path.append("/app")

from shared.schemas.base import AgentResponse, HealthResponse
from shared.middleware.auth import verify_service_token, audit_log_middleware
from shared.utils.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()
app = FastAPI(title="SmartBank AI — Data Aggregation Agent", version="1.0.0")
app.middleware("http")(audit_log_middleware)

_start_time = time.time()

# Canonical transaction schema
class CanonicalTransaction(BaseModel):
    transaction_id: str
    timestamp_utc: str
    amount_ngn: float
    currency: str = "NGN"
    channel: str
    sender_bvn: Optional[str] = None
    receiver_bvn: Optional[str] = None
    category: Optional[str] = None
    status: str
    source_system: str
    checksum: str


def generate_checksum(data: dict) -> str:
    payload = f"{data.get('transaction_id')}{data.get('amount_ngn')}{data.get('timestamp_utc')}"
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def normalise_finacle_csv_row(row: dict) -> CanonicalTransaction:
    """Normalise a Finacle CSV export row to canonical schema."""
    return CanonicalTransaction(
        transaction_id=row.get("TXN_REF", row.get("transaction_ref", "")),
        timestamp_utc=row.get("TXN_DATE", row.get("transaction_date", "")),
        amount_ngn=float(row.get("TXN_AMOUNT", row.get("amount", 0))),
        currency=row.get("CURRENCY", "NGN"),
        channel=row.get("CHANNEL", "unknown").lower(),
        sender_bvn=row.get("DEBIT_BVN", row.get("sender_bvn")),
        receiver_bvn=row.get("CREDIT_BVN", row.get("receiver_bvn")),
        category=row.get("TXN_TYPE", row.get("transaction_type")),
        status=row.get("STATUS", "completed").lower(),
        source_system="finacle",
        checksum=generate_checksum(row),
    )


def normalise_mobile_json(txn: dict) -> CanonicalTransaction:
    """Normalise a mobile app API JSON transaction."""
    return CanonicalTransaction(
        transaction_id=txn.get("id", txn.get("transactionId", "")),
        timestamp_utc=txn.get("createdAt", txn.get("timestamp", "")),
        amount_ngn=float(txn.get("amount", txn.get("amountNgn", 0))),
        currency="NGN",
        channel=txn.get("channel", "mobile").lower(),
        sender_bvn=txn.get("senderBvn", txn.get("sender", {}).get("bvn") if isinstance(txn.get("sender"), dict) else None),
        receiver_bvn=txn.get("receiverBvn", txn.get("receiver", {}).get("bvn") if isinstance(txn.get("receiver"), dict) else None),
        category=txn.get("category", txn.get("type")),
        status=txn.get("status", "completed").lower(),
        source_system="mobile_app",
        checksum=generate_checksum(txn),
    )


def normalise_nip_xml(xml_str: str) -> list[CanonicalTransaction]:
    """Normalise NIP interbank settlement XML."""
    results = []
    try:
        root = ET.fromstring(xml_str)
        for txn in root.findall(".//Transaction"):
            data = {
                "transaction_id": txn.findtext("SessionID", ""),
                "timestamp_utc": txn.findtext("TransactionTime", ""),
                "amount_ngn": float(txn.findtext("Amount", 0)),
                "channel": "nip",
                "sender_bvn": txn.findtext("OriginatorBVN"),
                "receiver_bvn": txn.findtext("BeneficiaryBVN"),
                "status": txn.findtext("ResponseCode", "00") == "00" and "completed" or "failed",
            }
            results.append(CanonicalTransaction(
                **data,
                currency="NGN",
                category="interbank_transfer",
                source_system="nip",
                checksum=generate_checksum(data),
            ))
    except ET.ParseError as e:
        logger.error("NIP XML parse error: %s", e)
    return results


class NormaliseRequest(BaseModel):
    source: str  # "finacle" | "mobile" | "nip"
    records: list[dict]


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        agent="data_aggregation", version="1.0.0",
        model_loaded=True, uptime_seconds=round(time.time() - _start_time, 1),
    )


@app.post("/normalise", response_model=AgentResponse, dependencies=[Depends(verify_service_token)])
async def normalise(req: NormaliseRequest):
    start = time.monotonic()
    normalised = []
    errors = []

    for i, record in enumerate(req.records):
        try:
            if req.source == "finacle":
                normalised.append(normalise_finacle_csv_row(record).model_dump())
            elif req.source == "mobile":
                normalised.append(normalise_mobile_json(record).model_dump())
            else:
                errors.append({"index": i, "error": f"Unknown source: {req.source}"})
        except Exception as e:
            errors.append({"index": i, "error": str(e)})

    latency = (time.monotonic() - start) * 1000
    return AgentResponse(
        agent="data_aggregation", version="1.0.0", latency_ms=round(latency, 2),
        payload={
            "source": req.source,
            "total_input": len(req.records),
            "normalised_count": len(normalised),
            "error_count": len(errors),
            "records": normalised,
            "errors": errors,
        },
    )
