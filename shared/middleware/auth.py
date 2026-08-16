"""
HMAC-based inter-service authentication middleware.
Every agent validates inbound requests from the orchestrator using this middleware.
"""
import hmac
import time
from fastapi import Request, HTTPException, status
from shared.utils.config import get_settings


async def verify_service_token(request: Request):
    """FastAPI dependency that validates the X-Service-Token header."""
    settings = get_settings()
    token = request.headers.get("X-Service-Token", "")

    if not settings.service_auth_token:
        # Token validation disabled in development if no token configured
        if settings.environment == "development":
            return
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Service auth token not configured",
        )

    if not hmac.compare_digest(token, settings.service_auth_token):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid service token",
        )


async def audit_log_middleware(request: Request, call_next):
    """Logs every prediction request for the HITL audit trail."""
    start = time.monotonic()
    response = await call_next(request)
    latency_ms = (time.monotonic() - start) * 1000

    # Structured log — consumed by the platform's audit pipeline
    import logging
    logger = logging.getLogger("audit")
    logger.info(
        "agent_invocation",
        extra={
            "path": request.url.path,
            "method": request.method,
            "status_code": response.status_code,
            "latency_ms": round(latency_ms, 2),
            "client": request.headers.get("X-Client-ID", "unknown"),
        },
    )
    return response
