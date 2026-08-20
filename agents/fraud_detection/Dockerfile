# ─────────────────────────────────────────────────────────────────────────────
# Agent 1: Fraud Detection Agent
# Port: 8002 | Model: LightGBM + SHAP
# ─────────────────────────────────────────────────────────────────────────────
FROM smartbank-ai-base:latest AS fraud-detection

# Agent-specific dependencies
COPY agents/fraud_detection/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Application code
COPY agents/fraud_detection/ /app/agent/

# Model artefacts (mounted as volume in production; baked in for staging)
USER root
RUN mkdir -p /app/models && chown smartbank:smartbank /app/models
USER smartbank

ENV AGENT_NAME=fraud_detection \
    AGENT_VERSION=1.0.0 \
    PORT=8002 \
    MODEL_DIR=/app/models \
    PYTHONPATH=/app

EXPOSE 8002

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8002/health || exit 1

CMD ["uvicorn", "agent.main:app", "--host", "0.0.0.0", "--port", "8002", "--workers", "2"]
