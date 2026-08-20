# ─────────────────────────────────────────────────────────────────────────────
# Agent 6: Conversational Ai Agent
# Port: 8007 | Model: local TF-IDF retrieval + Claude API
# ─────────────────────────────────────────────────────────────────────────────
FROM smartbank-ai-base:latest AS conversational-ai

# Agent-specific dependencies. No PyTorch: it was here for sentence-transformers,
# which the dependency hardening removed. Retrieval is a local TF-IDF baseline
# that needs nothing beyond scikit-learn.
COPY agents/conversational_ai/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Application code
# Mirror the repository layout. Serialised artefacts record the module path of
# any class they contain, so a container that flattens agents/<name> to agent/
# cannot unpickle its own models.
COPY agents/conversational_ai/ /app/agents/conversational_ai/

# Model artefacts directory

ENV AGENT_NAME=conversational_ai \
    AGENT_VERSION=1.0.0 \
    PORT=8007 \
    MODEL_DIR=/app/models \
    PYTHONPATH=/app

EXPOSE 8007

# Drop privileges for the running service.
USER smartbank

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8007/health || exit 1

CMD ["uvicorn", "agents.conversational_ai.main:app", "--host", "0.0.0.0", "--port", "8007", "--workers", "2"]
