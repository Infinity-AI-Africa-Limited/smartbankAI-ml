"""
Centralised configuration loader for all SmartBank AI agents.
Values are read from environment variables with sensible defaults.
"""
import os
from functools import lru_cache
from pydantic_settings import BaseSettings


class AgentSettings(BaseSettings):
    # Service identity
    agent_name: str = os.getenv("AGENT_NAME", "unknown")
    agent_version: str = os.getenv("AGENT_VERSION", "1.0.0")
    port: int = int(os.getenv("PORT", 8000))
    environment: str = os.getenv("ENVIRONMENT", "development")

    # Security
    service_auth_token: str = os.getenv("SERVICE_AUTH_TOKEN", "")
    orchestrator_url: str = os.getenv("ORCHESTRATOR_URL", "http://orchestrator:8001")
    orchestrator_allowed_client_id: str = os.getenv("ORCHESTRATOR_ALLOWED_CLIENT_ID", "smartbank-platform")
    contract_version: str = os.getenv("SMARTBANK_ML_CONTRACT_VERSION", "2026-08-01")

    # Database (for agents that persist predictions)
    database_url: str = os.getenv("DATABASE_URL", "")

    # Model artefacts
    model_dir: str = os.getenv("MODEL_DIR", "/app/models")

    # LLM (Conversational agent)
    anthropic_api_key: str = os.getenv("ANTHROPIC_API_KEY", "")
    llm_model: str = os.getenv("LLM_MODEL", "claude-3-5-sonnet-20241022")
    enable_remote_rag: bool = os.getenv("SMARTBANK_ENABLE_REMOTE_RAG", "false").lower() == "true"

    # Vector store (Conversational agent)
    chroma_host: str = os.getenv("CHROMA_HOST", "chromadb")
    chroma_port: int = int(os.getenv("CHROMA_PORT", 8000))

    # Observability
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    enable_mlflow: bool = os.getenv("ENABLE_MLFLOW", "false").lower() == "true"
    mlflow_tracking_uri: str = os.getenv("MLFLOW_TRACKING_URI", "")

    class Config:
        env_file = ".env"
        case_sensitive = False


@lru_cache()
def get_settings() -> AgentSettings:
    return AgentSettings()
