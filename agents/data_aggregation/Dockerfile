# ─────────────────────────────────────────────────────────────────────────────
# Agent 8: Data Aggregation Agent
# Port: 8009 | Model: ETL + Entity Resolution
# ─────────────────────────────────────────────────────────────────────────────
FROM smartbank-ai-base:latest AS data-aggregation

# Agent-specific dependencies
COPY agents/data_aggregation/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Application code
COPY agents/data_aggregation/ /app/agent/

# Model artefacts directory
RUN mkdir -p /app/models

ENV AGENT_NAME=data_aggregation \
    AGENT_VERSION=1.0.0 \
    PORT=8009 \
    MODEL_DIR=/app/models \
    PYTHONPATH=/app

EXPOSE 8009

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8009/health || exit 1

CMD ["uvicorn", "agent.main:app", "--host", "0.0.0.0", "--port", "8009", "--workers", "2"]
