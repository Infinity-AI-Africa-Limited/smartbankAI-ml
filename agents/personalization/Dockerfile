# ─────────────────────────────────────────────────────────────────────────────
# Agent 4: Personalization Agent
# Port: 8005 | Model: SVD Collaborative Filtering + LightGBM NBA
# ─────────────────────────────────────────────────────────────────────────────
FROM smartbank-ai-base:latest AS personalization

# Agent-specific dependencies
COPY agents/personalization/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Application code
COPY agents/personalization/ /app/agent/

# Model artefacts directory
RUN mkdir -p /app/models

ENV AGENT_NAME=personalization \
    AGENT_VERSION=1.0.0 \
    PORT=8005 \
    MODEL_DIR=/app/models \
    PYTHONPATH=/app

EXPOSE 8005

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8005/health || exit 1

CMD ["uvicorn", "agent.main:app", "--host", "0.0.0.0", "--port", "8005", "--workers", "2"]
