"""Validate local Compose model mount declarations without requiring Docker."""

from __future__ import annotations

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
COMPOSE_PATH = ROOT / "infra" / "docker" / "docker-compose.yml"
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


if __name__ == "__main__":
    validate()
    print("Compose model mount configuration is valid")
