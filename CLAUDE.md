# SmartBank AI ML Services — Claude Code Context

This repository is the ML-service layer of the SmartBank AI platform. Read `docs/CLAUDE_CODE_PRODUCTION_HANDOFF.md` first for the complete implementation history, security posture, model status, production gates, and review order.

## Cross-repository relationship

| Repository | Authoritative organisation repo | Personal mirror | Role |
|---|---|---|---|
| Application platform | `Infinity-AI-Africa-Limited/smartbankai-platform` | `MistaRichMan/smartbankai` | React/tRPC/Drizzle/MySQL platform, customer/tenant/admin portals, RBAC, approval workflow, and server-only ML gateway |
| ML services | `Infinity-AI-Africa-Limited/smartbankAI-ml` | `MistaRichMan/smartbankAI-ml` | Orchestrator, eight FastAPI agents, model services, containers, Kubernetes, and ML CI |

The browser must never call an agent directly. The platform backend calls the orchestrator under the `ml-orchestrator.v1` contract; the orchestrator makes authenticated private service calls to individual agents.

## Non-negotiable safety posture

- All outputs are **advisory** and must indicate human review is required for high-impact workflows.
- Do not allow agent responses to autonomously approve/decline credit, block/unblock a customer, submit an AML/SAR report, move funds, or modify KYC.
- Synthetic data and artefacts are development-only; they are not production validation evidence.
- Keep model/data artefacts out of Git. Use signed, versioned, protected production storage and a model registry.
- Never accept arbitrary prompt templates, arbitrary retrieval paths, untrusted tool configuration, or unrestricted outbound access.

## Active review sequence

1. Review dependency hardening: `MistaRichMan/smartbankAI-ml#1` / `Infinity-AI-Africa-Limited/smartbankAI-ml#2`.
2. Review synthetic models: `MistaRichMan/smartbankAI-ml#2` / `Infinity-AI-Africa-Limited/smartbankAI-ml#3`.
3. Review the platform’s server-only gateway, audit persistence, and tRPC integration in the companion platform repository.

## Required checks

```bash
./scripts/audit_dependencies.sh
ruff check agents orchestrator shared tests
pytest tests/unit -q
python3 -m compileall -q agents orchestrator shared
```

Before a controlled bank UAT, complete mTLS/workload identity, secrets rotation, tenant isolation, payload redaction, private network policies, image/SBOM/container scanning, independent model validation, human-approval testing, integration/negative-authorisation testing, and rollback/DR evidence.
