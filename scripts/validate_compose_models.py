"""Validate local Compose model mount declarations without requiring Docker."""

from __future__ import annotations

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
COMPOSE_PATH = ROOT / "infra" / "docker" / "docker-compose.yml"
DOCKERFILES = [*ROOT.glob("agents/*/Dockerfile"), ROOT / "orchestrator" / "Dockerfile"]
EXPECTED = {
    "fraud-detection": "fraud_detection",
    "credit-risk": "credit_risk",
    "aml-compliance": "aml_compliance",
    "personalization": "personalization",
    "predictive-analytics": "predictive_analytics",
    "conversational-ai": "conversational_ai",
    "smart-dashboard": "smart_dashboard",
    "data-aggregation": "data_aggregation",
}


def validate() -> None:
    compose = yaml.safe_load(COMPOSE_PATH.read_text(encoding="utf-8"))
    for service, definition in compose["services"].items():
        image = definition.get("image")
        if image and image != image.lower():
            raise ValueError(f"{service} has an invalid uppercase Docker image tag: {image}")
    for service, agent in EXPECTED.items():
        definition = compose["services"][service]
        mounts = definition.get("volumes", [])
        expected_mount = f"../../agents/{agent}/models:/app/models:ro"
        if expected_mount not in mounts:
            raise ValueError(f"{service} is missing read-only model mount {expected_mount}")
        if definition.get("environment", {}).get("MODEL_DIR") != "/app/models":
            raise ValueError(f"{service} is missing MODEL_DIR=/app/models")
    for dockerfile in DOCKERFILES:
        contents = dockerfile.read_text(encoding="utf-8")
        if "smartbankAI-base" in contents:
            raise ValueError(f"{dockerfile.relative_to(ROOT)} references an invalid uppercase base image")
    for service in EXPECTED:
        dockerfile = ROOT / "agents" / EXPECTED[service] / "Dockerfile"
        contents = dockerfile.read_text(encoding="utf-8")
        required_model_setup = "USER root\nRUN mkdir -p /app/models && chown smartbank:smartbank /app/models\nUSER smartbank"
        if required_model_setup not in contents:
            raise ValueError(f"{dockerfile.relative_to(ROOT)} does not initialise /app/models for the non-root runtime user")
    conversational_dockerfile = ROOT / "agents" / "conversational_ai" / "Dockerfile"
    conversational_contents = conversational_dockerfile.read_text(encoding="utf-8")
    required_cpu_torch = "--index-url https://download.pytorch.org/whl/cpu torch==2.13.0+cpu"
    if required_cpu_torch not in conversational_contents:
        raise ValueError("The conversational Dockerfile must pin the CPU-only PyTorch wheel for local Compose builds")


if __name__ == "__main__":
    validate()
    print("Compose model mount configuration is valid")
