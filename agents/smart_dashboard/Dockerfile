# ─────────────────────────────────────────────────────────────────────────────
# Agent 7: Smart Dashboard Agent
# Port: 8008 | Model: K-Means + NLG
# ─────────────────────────────────────────────────────────────────────────────
FROM smartbank-ai-base:latest AS smart-dashboard

# Agent-specific dependencies
COPY agents/smart_dashboard/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Application code
# Mirror the repository layout. Serialised artefacts record the module path of
# any class they contain, so a container that flattens agents/<name> to agent/
# cannot unpickle its own models.
COPY agents/smart_dashboard/ /app/agents/smart_dashboard/

# Model artefacts directory

ENV AGENT_NAME=smart_dashboard \
    AGENT_VERSION=1.0.0 \
    PORT=8008 \
    MODEL_DIR=/app/models \
    PYTHONPATH=/app

EXPOSE 8008

# Drop privileges for the running service.
USER smartbank

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8008/health || exit 1

CMD ["uvicorn", "agents.smart_dashboard.main:app", "--host", "0.0.0.0", "--port", "8008", "--workers", "2"]
