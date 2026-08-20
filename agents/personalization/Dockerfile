# ─────────────────────────────────────────────────────────────────────────────
# Agent 4: Personalization Agent
# Port: 8005 | Model: SVD Collaborative Filtering + LightGBM NBA
# ─────────────────────────────────────────────────────────────────────────────
FROM smartbank-ai-base:latest AS personalization

# Agent-specific dependencies
COPY agents/personalization/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Application code
# Mirror the repository layout. Serialised artefacts record the module path of
# any class they contain, so a container that flattens agents/<name> to agent/
# cannot unpickle its own models.
COPY agents/personalization/ /app/agents/personalization/

# Model artefacts directory

ENV AGENT_NAME=personalization \
    AGENT_VERSION=1.0.0 \
    PORT=8005 \
    MODEL_DIR=/app/models \
    PYTHONPATH=/app

EXPOSE 8005

# Drop privileges for the running service.
USER smartbank

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8005/health || exit 1

CMD ["uvicorn", "agents.personalization.main:app", "--host", "0.0.0.0", "--port", "8005", "--workers", "2"]
