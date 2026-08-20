# ─────────────────────────────────────────────────────────────────────────────
# SmartBank AI — Agent Orchestrator
# Port: 8001 | Role: Central router, circuit breaker, audit logger
# ─────────────────────────────────────────────────────────────────────────────
FROM smartbank-ai-base:latest AS orchestrator

COPY orchestrator/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY orchestrator/ /app/orchestrator/

ENV AGENT_NAME=orchestrator \
    AGENT_VERSION=1.0.0 \
    PORT=8001 \
    PYTHONPATH=/app

EXPOSE 8001

# Drop privileges for the running service.
USER smartbank

HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1

CMD ["uvicorn", "orchestrator.main:app", "--host", "0.0.0.0", "--port", "8001", "--workers", "4"]
