# SmartBank AI Synthetic Model Build Runbook

## Purpose

This runbook builds the development-only synthetic datasets and artefacts for all eight SmartBank AI agents. The generated data and models validate pipeline behaviour, service contracts, explainability surfaces, and Human-in-the-Loop workflows. They are not production models and must not be used for lending, fraud blocking, AML/NFIU filing, customer targeting, or any autonomous banking action.

## Local build

From the ML repository root, install the agent requirements in a controlled Python environment and run:

```bash
./scripts/build_synthetic_models.sh
ruff check agents shared scripts tests
pytest tests/unit -q
```

The command creates `data/synthetic/` and per-agent `agents/<agent>/models/` directories. Both locations are intentionally ignored by Git because they are regenerated from the fixed seed `20260816`.

## Artefacts

| Agent | Key artefacts |
|---|---|
| Fraud Detection | `fraud_lgbm.pkl`, `fraud_tree_shap_samples.json`, evaluation report, model card |
| Credit Risk | `credit_scorecard.pkl`, `credit_lgbm.pkl`, evaluation report, model card |
| AML / Compliance | `aml_graph_isolation.pkl`, rules configuration, graph scores, evaluation report, model card |
| Personalization | next-best-action model, K-means model, product similarity matrix, segment file, model card |
| Predictive Analytics | churn, cash-flow, and autoregressive volume models, evaluation report, model card |
| Conversational AI | TF-IDF retrieval baseline, retrieval evaluation report, model card |
| Smart Dashboard | dashboard segment model, segment profiles, deterministic insight snapshot, model card |
| Data Aggregation | canonical fixture output, entity-match model, normalisation report, model card |

## Local service verification

The local Compose configuration mounts each generated `models/` directory **read-only** into the relevant service as `/app/models`.

```bash
docker compose -f infra/docker/docker-compose.yml up --build
./scripts/health_check.sh
```

Every score, recommendation, alert, forecast, retrieval response, normalisation result, and dashboard output must be surfaced as an advisory. The platform gateway and user interface must require human review before high-impact action.

## Production-replacement gate

Before a bank environment receives a trained model, replace the synthetic dataset with approved, privacy-governed historical data. The institution’s independent model-risk function must validate performance, calibration, explainability, stability, fairness, drift thresholds, access controls, and the Human-in-the-Loop policy. Register the approved model version separately from source code, deploy it through a protected artefact store, and retain rollback and audit evidence.
