# ─────────────────────────────────────────────────────────────────────────────
# Agent 6: Conversational Ai Agent
# Port: 8007 | Model: RAG + Claude API
# ─────────────────────────────────────────────────────────────────────────────
FROM smartbank-ai-base:latest AS conversational-ai

# Agent-specific dependencies
COPY agents/conversational_ai/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Application code
COPY agents/conversational_ai/ /app/agent/

# Model artefacts directory
RUN mkdir -p /app/models

ENV AGENT_NAME=conversational_ai \
    AGENT_VERSION=1.0.0 \
    PORT=8007 \
    MODEL_DIR=/app/models \
    PYTHONPATH=/app

EXPOSE 8007

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8007/health || exit 1

CMD ["uvicorn", "agent.main:app", "--host", "0.0.0.0", "--port", "8007", "--workers", "2"]
