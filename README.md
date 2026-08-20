# SmartBank AI — ML Agent Services Repository

**Infinity AI Africa Limited × SmartBank AI**

This repository contains the machine learning agent services that power the SmartBank AI platform. It is structured as a monorepo where each of the 8 AI agents is an independently deployable Python microservice, coordinated by a central orchestrator.

---

## Repository Structure

```
smartbankAI-ml/
├── agents/
│   ├── fraud_detection/          # Agent 1 — LightGBM fraud classifier + SHAP
│   ├── credit_risk/              # Agent 2 — WoE scorecard + LightGBM PD model
│   ├── aml_compliance/           # Agent 3 — Rule engine + NetworkX graph analysis
│   ├── personalization/          # Agent 4 — Collaborative filtering + NBA model
│   ├── predictive_analytics/     # Agent 5 — Prophet + ARIMA + churn prediction
│   ├── conversational_ai/        # Agent 6 — RAG pipeline + LLM orchestration
│   ├── smart_dashboard/          # Agent 7 — Clustering + NLG insight generator
│   └── data_aggregation/         # Agent 8 — ETL normalisation + entity resolution
├── orchestrator/                 # Central agent router with circuit breaker
├── shared/
│   ├── schemas/                  # Pydantic request/response schemas (shared)
│   ├── utils/                    # Logging, auth, config helpers
│   └── middleware/               # Auth middleware, rate limiting, audit logging
├── infra/
│   ├── docker/                   # Shared base images
│   ├── k8s/                      # Kubernetes manifests (base + overlays)
│   ├── terraform/                # Cloud infrastructure as code
│   └── helm/                     # Helm chart for full stack deployment
├── ci/                           # GitHub Actions CI/CD workflows
├── scripts/                      # Dev setup, data loading, model training scripts
├── docs/                         # Architecture diagrams, model cards, runbooks
└── tests/                        # Unit, integration, and e2e test suites
```

---

## Architecture Overview

```
SmartBank AI Platform (Node.js / tRPC)
              │
              ▼
   Agent Orchestrator  :8001
   (FastAPI — routes, circuit breaker, audit log)
              │
   ┌──────────┼──────────────────────────────────┐
   │          │          │          │            │
   ▼          ▼          ▼          ▼            ▼
Fraud      Credit      AML     Personali-   Predictive
:8002       :8003      :8004    zation:8005  Analytics:8006
                                             │
                                   Conversational:8007
                                   Dashboard:8008
                                   DataAgg:8009
```

All agents expose a `/health`, `/predict` (or `/recommend`, `/analyse`, etc.), and `/explain` endpoint. The orchestrator is the only service the tRPC platform calls directly.

---

## Quick Start (Local Development)

```bash
# 1. Clone the repository
git clone https://github.com/infinityai-africa/smartbankAI-ml.git
cd smartbankAI-ml

# 2. Copy environment template
cp .env.example .env
# Edit .env with your values

# 3. Start all services with Docker Compose
docker compose -f infra/docker/docker-compose.yml up --build

# 4. Verify all agents are healthy
./scripts/health_check.sh
```

---

## Deployment

| Environment | Command |
|---|---|
| Local dev | `docker compose up --build` |
| Staging | `kubectl apply -k infra/k8s/overlays/staging` |
| Production | `kubectl apply -k infra/k8s/overlays/production` |
| Full cloud (Terraform) | `cd infra/terraform && terraform apply` |

---

## Synthetic Model Build (Development Only)

The repository includes a reproducible, privacy-safe synthetic Nigerian banking corpus and a full training pipeline for all eight agents. The generated datasets and artefacts are **development-only** and must never be used for production banking decisions.

```bash
./scripts/build_synthetic_models.sh
ruff check agents shared scripts tests
pytest tests/unit -q
```

See [`docs/synthetic-training-runbook.md`](docs/synthetic-training-runbook.md) for artefacts, local service mounting, and mandatory real-data validation gates.

## Model Training

Each agent has a `train.py` script. To retrain a specific agent:

```bash
cd agents/fraud_detection
python train.py --data-path /data/transactions.csv --output-dir ./models
```

See `docs/synthetic-training-runbook.md` for the synthetic development workflow. A future fine-tuned conversational model requires a separately governed GPU training environment and an approved real banking corpus.

---

## Testing

```bash
# Unit tests
pytest tests/unit/ -v

# Integration tests (requires running services)
pytest tests/integration/ -v

# End-to-end tests
pytest tests/e2e/ -v
```

---

## Security

All inter-service communication is authenticated via a shared HMAC token (configured in `.env`). The orchestrator validates the token on every inbound request from the tRPC platform. Agent-to-agent calls within the cluster use mTLS in production.

See `docs/security_architecture.md` for the full threat model and CBN compliance mapping.

---

## Licence

Proprietary — Infinity AI Africa Limited. All rights reserved.
