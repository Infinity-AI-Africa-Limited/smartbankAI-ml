# ─────────────────────────────────────────────────────────────────────────────
# Agent 3: Aml Compliance Agent
# Port: 8004 | Model: Rule Engine + NetworkX
# ─────────────────────────────────────────────────────────────────────────────
FROM smartbank-ai-base:latest AS aml-compliance

# Agent-specific dependencies
COPY agents/aml_compliance/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Application code
# Mirror the repository layout. Serialised artefacts record the module path of
# any class they contain, so a container that flattens agents/<name> to agent/
# cannot unpickle its own models.
COPY agents/aml_compliance/ /app/agents/aml_compliance/

# Model artefacts directory

ENV AGENT_NAME=aml_compliance \
    AGENT_VERSION=1.0.0 \
    PORT=8004 \
    MODEL_DIR=/app/models \
    PYTHONPATH=/app

EXPOSE 8004

# Drop privileges for the running service.
USER smartbank

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8004/health || exit 1

CMD ["uvicorn", "agents.aml_compliance.main:app", "--host", "0.0.0.0", "--port", "8004", "--workers", "2"]
