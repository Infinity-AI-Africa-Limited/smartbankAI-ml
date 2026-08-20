"""
Inter-service authentication middleware.
Every agent validates inbound requests from the orchestrator using this middleware.
"""
import hmac
import logging
import time
from fastapi import Request, HTTPException, status
from shared.utils.config import get_settings

logger = logging.getLogger(__name__)

MIN_TOKEN_LENGTH = 32


def require_secure_configuration() -> None:
    """Refuse to start when the service token is missing or too weak.

    Called from each service's startup hook. There is no development bypass: a
    service that cannot authenticate its callers must not accept traffic in any
    environment, because the environment marker is itself configuration and can
    be wrong.
    """
    settings = get_settings()
    token = settings.service_auth_token
    if not token:
        raise RuntimeError(
            "SERVICE_AUTH_TOKEN is not configured. Every SmartBank ML service requires a "
            "service token; there is no unauthenticated mode."
        )
    if len(token) < MIN_TOKEN_LENGTH:
        raise RuntimeError(
            f"SERVICE_AUTH_TOKEN must be at least {MIN_TOKEN_LENGTH} characters; "
            f"got {len(token)}."
        )


async def verify_service_token(request: Request):
    """FastAPI dependency that validates the X-Service-Token header.

    Fails closed. A missing or under-length configured token yields 503 rather
    than silently accepting the request.
    """
    settings = get_settings()
    configured = settings.service_auth_token

    if not configured or len(configured) < MIN_TOKEN_LENGTH:
        logger.error("Service auth token is not configured or is too short; rejecting request")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service authentication is not configured",
        )

    presented = request.headers.get("X-Service-Token", "")
    if not hmac.compare_digest(presented, configured):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid service token",
        )


async def audit_log_middleware(request: Request, call_next):
    """Logs every prediction request for the HITL audit trail.

    Records routing and timing metadata only. Request and response bodies are
    never logged: they may carry customer data that must not reach a developer
    log under the platform's data-protection contract.
    """
    start = time.monotonic()
    response = await call_next(request)
    latency_ms = (time.monotonic() - start) * 1000

    audit = logging.getLogger("audit")
    audit.info(
        "agent_invocation",
        extra={
            "path": request.url.path,
            "method": request.method,
            "status_code": response.status_code,
            "latency_ms": round(latency_ms, 2),
            "client": request.headers.get("X-Client-ID", "unknown"),
            "correlation_id": request.headers.get("X-Correlation-ID", "unset"),
        },
    )
    return response
