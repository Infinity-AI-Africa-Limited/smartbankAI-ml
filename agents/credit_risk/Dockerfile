# ─────────────────────────────────────────────────────────────────────────────
# Agent 2: Credit Risk Agent
# Port: 8003 | Model: WoE Scorecard + LightGBM
# ─────────────────────────────────────────────────────────────────────────────
FROM smartbank-ai-base:latest AS credit-risk

# Agent-specific dependencies
COPY agents/credit_risk/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Application code
# Mirror the repository layout. Serialised artefacts record the module path of
# any class they contain, so a container that flattens agents/<name> to agent/
# cannot unpickle its own models.
COPY agents/credit_risk/ /app/agents/credit_risk/

# Model artefacts directory

ENV AGENT_NAME=credit_risk \
    AGENT_VERSION=1.0.0 \
    PORT=8003 \
    MODEL_DIR=/app/models \
    PYTHONPATH=/app

EXPOSE 8003

# Drop privileges for the running service.
USER smartbank

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8003/health || exit 1

CMD ["uvicorn", "agents.credit_risk.main:app", "--host", "0.0.0.0", "--port", "8003", "--workers", "2"]
