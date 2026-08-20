"""Deterministic safety guardrails for SmartBank AI advisory conversations."""

from __future__ import annotations


def safety_response(message: str) -> str | None:
    lower = message.lower()
    if any(term in lower for term in ["another customer", "someone else's", "somebody else's"]):
        return "I cannot provide another customer's information. Please contact an authorised bank representative for help with your own account."
    if ("transfer" in lower or "send money" in lower) and any(term in lower for term in ["automatically", "without asking", "without confirmation"]):
        return "I cannot execute or block a transfer without your confirmation and human review. I can prepare an advisory recommendation for your review."
    if "loan" in lower and any(term in lower for term in ["approve", "skip", "without"]):
        return "I cannot approve or decline a loan. Officer review is required before a qualified credit officer can assess the application and supporting evidence."
    if any(term in lower for term in ["aml report", "sar", "compliance report"]) and any(term in lower for term in ["file", "submit", "without"]):
        return "I cannot file a compliance report. A qualified Compliance Officer must review the evidence and follow the institution's approved filing workflow."
    return None
