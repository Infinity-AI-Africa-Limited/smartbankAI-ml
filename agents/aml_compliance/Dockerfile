# ─────────────────────────────────────────────────────────────────────────────
# Agent 3: Aml Compliance Agent
# Port: 8004 | Model: Rule Engine + NetworkX
# ─────────────────────────────────────────────────────────────────────────────
FROM smartbank-ai-base:latest AS aml-compliance

# Agent-specific dependencies
COPY agents/aml_compliance/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Application code
COPY agents/aml_compliance/ /app/agent/

# Model artefacts directory
RUN mkdir -p /app/models

ENV AGENT_NAME=aml_compliance \
    AGENT_VERSION=1.0.0 \
    PORT=8004 \
    MODEL_DIR=/app/models \
    PYTHONPATH=/app

EXPOSE 8004

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8004/health || exit 1

CMD ["uvicorn", "agent.main:app", "--host", "0.0.0.0", "--port", "8004", "--workers", "2"]
