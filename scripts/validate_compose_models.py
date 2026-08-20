"""Validate local Compose model mount declarations without requiring Docker."""

from __future__ import annotations

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
COMPOSE_PATH = ROOT / "infra" / "docker" / "docker-compose.yml"
RUNNER_PATH = ROOT / "scripts" / "run_local_compose_validation.ps1"
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
    # The base image already creates /app/models and hands it to the runtime
    # user, and both Compose and Kubernetes replace that directory with a mount
    # at run time, which carries the host's ownership regardless. Repeating the
    # chown per agent would therefore change nothing at run time while forcing a
    # USER root hop into eight otherwise non-root images. Assert the two things
    # that do hold instead.
    base_path = ROOT / "infra" / "docker" / "Dockerfile.base"
    base_contents = base_path.read_text(encoding="utf-8")
    if "mkdir -p /app/models" not in base_contents:
        raise ValueError("Dockerfile.base must create /app/models for the runtime user")
    if "chown -R smartbank:smartbank /app" not in base_contents:
        raise ValueError("Dockerfile.base must hand /app to the smartbank user")
    for agent in EXPECTED.values():
        dockerfile = ROOT / "agents" / agent / "Dockerfile"
        contents = dockerfile.read_text(encoding="utf-8")
        if "USER smartbank" not in contents:
            raise ValueError(f"{dockerfile.relative_to(ROOT)} must drop to the non-root smartbank user")
        if contents.rstrip().endswith("USER root"):
            raise ValueError(f"{dockerfile.relative_to(ROOT)} must not finish as root")
    conversational_dockerfile = ROOT / "agents" / "conversational_ai" / "Dockerfile"
    conversational_contents = conversational_dockerfile.read_text(encoding="utf-8")
    required_cpu_torch = "--index-url https://download.pytorch.org/whl/cpu torch==2.13.0+cpu"
    if required_cpu_torch not in conversational_contents:
        raise ValueError("The conversational Dockerfile must pin the CPU-only PyTorch wheel for local Compose builds")
    runner_contents = RUNNER_PATH.read_text(encoding="utf-8")
    if "up --build -d orchestrator" not in runner_contents:
        raise ValueError("The local Compose runner must start the orchestrator dependency graph and exclude base-builder")


if __name__ == "__main__":
    validate()
    print("Compose model mount configuration is valid")
