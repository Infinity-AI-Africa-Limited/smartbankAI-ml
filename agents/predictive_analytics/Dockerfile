# ─────────────────────────────────────────────────────────────────────────────
# Agent 5: Predictive Analytics Agent
# Port: 8006 | Model: Prophet + LightGBM Churn + ARIMA
# ─────────────────────────────────────────────────────────────────────────────
FROM smartbank-ai-base:latest AS predictive-analytics

# Agent-specific dependencies
COPY agents/predictive_analytics/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Application code
COPY agents/predictive_analytics/ /app/agent/

# Model artefacts directory

ENV AGENT_NAME=predictive_analytics \
    AGENT_VERSION=1.0.0 \
    PORT=8006 \
    MODEL_DIR=/app/models \
    PYTHONPATH=/app

EXPOSE 8006

# Drop privileges for the running service.
USER smartbank

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8006/health || exit 1

CMD ["uvicorn", "agent.main:app", "--host", "0.0.0.0", "--port", "8006", "--workers", "2"]
