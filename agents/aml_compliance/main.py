"""
Agent 3: AML Compliance Agent
Components: CBN typology rule engine + NetworkX graph analysis
Port: 8004
"""
import time
import logging
from datetime import datetime, timedelta
from collections import defaultdict
from fastapi import FastAPI, Depends
from pydantic import BaseModel
from typing import Optional
import sys
sys.path.append("/app")

from shared.schemas.base import AgentResponse, HealthResponse, RiskLevel
from shared.middleware.auth import verify_service_token, audit_log_middleware
from shared.utils.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()
app = FastAPI(title="SmartBank AI — AML Compliance Agent", version="1.0.0")
app.middleware("http")(audit_log_middleware)

_start_time = time.time()

# CBN structuring threshold (transactions just below ₦1,000,000)
STRUCTURING_THRESHOLD_NGN = 1_000_000
STRUCTURING_WINDOW_HOURS = 24
STRUCTURING_MIN_COUNT = 3

LAYERING_HOPS = 3
LAYERING_WINDOW_HOURS = 48

SMURFING_MIN_SENDERS = 5
SMURFING_WINDOW_DAYS = 7


class AMLRequest(BaseModel):
    customer_id: str
    transactions: list[dict]  # list of {id, sender, receiver, amount_ngn, timestamp}
    check_types: list[str] = ["structuring", "layering", "smurfing"]


class AMLResponse(BaseModel):
    customer_id: str
    risk_score: float
    risk_level: RiskLevel
    typologies_matched: list[str]
    sar_required: bool
    sar_narrative: Optional[str] = None
    flagged_transactions: list[str]


def check_structuring(txns: list[dict], customer_id: str) -> tuple[bool, list[str]]:
    """Detect multiple transactions just below ₦1M within 24 hours."""
    flagged = []
    cutoff = datetime.utcnow() - timedelta(hours=STRUCTURING_WINDOW_HOURS)
    recent = [
        t for t in txns
        if t["sender"] == customer_id
        and t["amount_ngn"] < STRUCTURING_THRESHOLD_NGN
        and t["amount_ngn"] > STRUCTURING_THRESHOLD_NGN * 0.8
        and datetime.fromisoformat(t["timestamp"]) > cutoff
    ]
    if len(recent) >= STRUCTURING_MIN_COUNT:
        flagged = [t["id"] for t in recent]
    return len(flagged) > 0, flagged


def check_smurfing(txns: list[dict], customer_id: str) -> tuple[bool, list[str]]:
    """Detect same beneficiary receiving from many different senders."""
    cutoff = datetime.utcnow() - timedelta(days=SMURFING_WINDOW_DAYS)
    inbound = [
        t for t in txns
        if t["receiver"] == customer_id
        and datetime.fromisoformat(t["timestamp"]) > cutoff
    ]
    unique_senders = set(t["sender"] for t in inbound)
    flagged = [t["id"] for t in inbound] if len(unique_senders) >= SMURFING_MIN_SENDERS else []
    return len(flagged) > 0, flagged


def check_layering(txns: list[dict], customer_id: str) -> tuple[bool, list[str]]:
    """Detect rapid fund movement through multiple accounts (simplified graph check)."""
    cutoff = datetime.utcnow() - timedelta(hours=LAYERING_WINDOW_HOURS)
    recent = [
        t for t in txns
        if datetime.fromisoformat(t["timestamp"]) > cutoff
    ]
    # Build adjacency: who sent to whom
    graph = defaultdict(set)
    for t in recent:
        graph[t["sender"]].add(t["receiver"])

    # BFS from customer_id — check if funds reach 3+ hops
    visited = {customer_id}
    frontier = {customer_id}
    hops = 0
    while frontier and hops < LAYERING_HOPS:
        next_frontier = set()
        for node in frontier:
            for neighbour in graph.get(node, set()):
                if neighbour not in visited:
                    visited.add(neighbour)
                    next_frontier.add(neighbour)
        frontier = next_frontier
        hops += 1

    is_layering = hops >= LAYERING_HOPS and len(visited) > LAYERING_HOPS
    flagged = [t["id"] for t in recent if t["sender"] == customer_id] if is_layering else []
    return is_layering, flagged


def generate_sar_narrative(customer_id: str, typologies: list[str], flagged_ids: list[str]) -> str:
    date_str = datetime.utcnow().strftime("%d %B %Y")
    typology_text = " and ".join(typologies)
    return (
        f"SUSPICIOUS ACTIVITY REPORT — {date_str}\n\n"
        f"Subject Account: {customer_id}\n"
        f"Reporting Institution: SmartBank AI (powered by Infinity AI Africa Limited)\n\n"
        f"Nature of Suspicion: The subject account has been flagged for {typology_text} "
        f"based on automated transaction pattern analysis conducted in accordance with "
        f"the Money Laundering (Prevention and Prohibition) Act 2022 and NFIU guidelines.\n\n"
        f"Flagged Transaction IDs: {', '.join(flagged_ids[:10])}{'...' if len(flagged_ids) > 10 else ''}\n\n"
        f"This report is submitted for review by the Compliance Officer prior to filing "
        f"with the Nigerian Financial Intelligence Unit (NFIU)."
    )


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        agent="aml_compliance", version="1.0.0",
        model_loaded=True, uptime_seconds=round(time.time() - _start_time, 1),
    )


@app.post("/analyse", response_model=AgentResponse, dependencies=[Depends(verify_service_token)])
async def analyse(req: AMLRequest):
    start = time.monotonic()
    typologies_matched = []
    all_flagged = []

    if "structuring" in req.check_types:
        matched, flagged = check_structuring(req.transactions, req.customer_id)
        if matched:
            typologies_matched.append("structuring")
            all_flagged.extend(flagged)

    if "smurfing" in req.check_types:
        matched, flagged = check_smurfing(req.transactions, req.customer_id)
        if matched:
            typologies_matched.append("smurfing")
            all_flagged.extend(flagged)

    if "layering" in req.check_types:
        matched, flagged = check_layering(req.transactions, req.customer_id)
        if matched:
            typologies_matched.append("layering")
            all_flagged.extend(flagged)

    risk_score = min(1.0, len(typologies_matched) * 0.4 + len(set(all_flagged)) * 0.02)
    risk_level = (
        RiskLevel.CRITICAL if risk_score >= 0.8 else
        RiskLevel.HIGH if risk_score >= 0.5 else
        RiskLevel.MEDIUM if risk_score >= 0.2 else
        RiskLevel.LOW
    )
    sar_required = risk_score >= 0.5
    sar_narrative = generate_sar_narrative(req.customer_id, typologies_matched, list(set(all_flagged))) if sar_required else None

    latency = (time.monotonic() - start) * 1000
    return AgentResponse(
        agent="aml_compliance", version="1.0.0", latency_ms=round(latency, 2),
        payload=AMLResponse(
            customer_id=req.customer_id,
            risk_score=round(risk_score, 4),
            risk_level=risk_level,
            typologies_matched=typologies_matched,
            sar_required=sar_required,
            sar_narrative=sar_narrative,
            flagged_transactions=list(set(all_flagged)),
        ).model_dump(),
    )
