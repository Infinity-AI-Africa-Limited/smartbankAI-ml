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
    # /app/models is created and chowned once in the base image, before any
    # derived build drops privileges. Each agent only drops to the runtime user.
    base_contents = (ROOT / 'infra' / 'docker' / 'Dockerfile.base').read_text(encoding='utf-8')
    if 'RUN mkdir -p /app/models && chown -R smartbank:smartbank /app' not in base_contents:
        raise ValueError('Dockerfile.base must create /app/models and hand it to the runtime user')
    for service in EXPECTED:
        dockerfile = ROOT / 'agents' / EXPECTED[service] / 'Dockerfile'
        contents = dockerfile.read_text(encoding='utf-8')
        if 'USER smartbank' not in contents:
            raise ValueError(f'{dockerfile.relative_to(ROOT)} does not drop to the non-root runtime user')
        # Artefacts record the module path of any class they contain, so the
        # container layout must match the repository layout or the agent cannot
        # unpickle its own models.
        agent = EXPECTED[service]
        if f'/app/agents/{agent}/' not in contents:
            raise ValueError(f'{dockerfile.relative_to(ROOT)} must copy the package to /app/agents/<name>/')
        if f'agents.{agent}.main:app' not in contents:
            raise ValueError(f'{dockerfile.relative_to(ROOT)} must serve agents.<name>.main:app')
    # Nothing imports torch. It was installed for sentence-transformers, which the
    # dependency hardening removed; reinstating it would add a large unused
    # dependency surface to a bank-facing image.
    conversational_contents = (ROOT / 'agents' / 'conversational_ai' / 'Dockerfile').read_text(encoding='utf-8')
    if 'torch' in conversational_contents:
        raise ValueError('The conversational Dockerfile must not install PyTorch; no module imports it')
    runner_contents = RUNNER_PATH.read_text(encoding="utf-8")
    if "up --build -d orchestrator" not in runner_contents:
        raise ValueError("The local Compose runner must start the orchestrator dependency graph and exclude base-builder")


if __name__ == "__main__":
    validate()
    print("Compose model mount configuration is valid")
