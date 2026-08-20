# ─────────────────────────────────────────────────────────────────────────────
# Agent 1: Fraud Detection Agent
# Port: 8002 | Model: LightGBM + SHAP
# ─────────────────────────────────────────────────────────────────────────────
FROM smartbank-ai-base:latest AS fraud-detection

# Agent-specific dependencies
COPY agents/fraud_detection/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Application code
# Mirror the repository layout. Serialised artefacts record the module path of
# any class they contain, so a container that flattens agents/<name> to agent/
# cannot unpickle its own models.
COPY agents/fraud_detection/ /app/agents/fraud_detection/

# Model artefacts (mounted as volume in production; baked in for staging)

ENV AGENT_NAME=fraud_detection \
    AGENT_VERSION=1.0.0 \
    PORT=8002 \
    MODEL_DIR=/app/models \
    PYTHONPATH=/app

EXPOSE 8002

# Drop privileges for the running service.
USER smartbank

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8002/health || exit 1

CMD ["uvicorn", "agents.fraud_detection.main:app", "--host", "0.0.0.0", "--port", "8002", "--workers", "2"]
