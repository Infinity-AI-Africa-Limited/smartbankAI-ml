# SmartBank AI — Claude Code Production-Hardening Handoff

**Owner:** Infinity AI Africa Limited
**Product:** SmartBank AI
**Prepared for:** Claude Code production review
**Status:** Prototype and synthetic-model build complete; **do not merge or deploy to a bank production environment without completing the gates in this document.**

## 1. Executive brief

SmartBank AI is an AI-native digital-banking SaaS platform designed for Nigerian financial institutions. Its market architecture is deliberately **human-in-the-loop**: the system may observe, score, explain, and recommend, but high-impact banking decisions must remain attributable to authorised bank staff. The product is therefore structured as a platform layer plus an independently deployable ML-service layer.

| Layer | Repository | Responsibility |
|---|---|---|
| SmartBank application platform | `Infinity-AI-Africa-Limited/smartbankai-platform` | React 19 portals, Node.js/tRPC backend, Drizzle/MySQL data model, RBAC, tenant workflows, audit views, and human approval controls |
| ML services | `Infinity-AI-Africa-Limited/smartbankAI-ml` | Orchestrator plus eight FastAPI agent services, model training/serving, container/Kubernetes scaffolding, and ML-specific CI |

The organisation repositories are the source of truth. The corresponding `MistaRichMan` repositories are working mirrors used for Claude Code review and collaboration.

> **Non-negotiable operating posture:** every result from the ML layer is an advisory recommendation. The application layer owns human approval, customer consent, audit retention, and any eventual action against bank data or customer accounts.

## 2. Implementation history and present state

The platform is already built as four connected portals: Infinity AI Super-Admin (`/dashboard`), SmartBank tenant operations (`/tenant/*`), customer Web Banking (`/banking/*`), and the mobile super-app (`/app/*`). It uses React 19, TypeScript, Tailwind, tRPC, Drizzle, and MySQL. The platform includes three roles—platform owner, tenant admin, analyst—and realistic Nigerian demo structures such as Naira amounts, BVN/NIN fields, Lagos/Abuja context, CBN compliance views, fraud alerts, transactions, credit applications, and AML artefacts.

The platform backend has been extended with a **server-only ML gateway** and a versioned orchestrator contract. The browser never calls an agent directly. The gateway minimises payloads, enforces timeouts and a circuit breaker, records immutable AI advisory evidence, and returns human-review-required recommendations. It supports fraud, credit, AML, recommendation, and assistant workflows through the orchestrator.

The ML repository contains eight services: Conversational AI, Fraud Detection, Credit Risk, Personalization, Predictive Analytics, Compliance/AML, Data Aggregation, and Smart Dashboard. The orchestrator is the only intended platform integration point.

## 3. Review order and pull requests

Claude Code should review the changes in this order. Keep the PRs separate; do not squash the model and security changes into one unreviewable deployment.

| Order | Workstream | Personal mirror PR | Organisation mirror PR | Review intent |
|---|---|---|---|---|
| 1 | Dependency hardening | https://github.com/MistaRichMan/smartbankAI-ml/pull/1 | https://github.com/Infinity-AI-Africa-Limited/smartbankAI-ml/pull/2 | Establish a secure dependency baseline before model work |
| 2 | Synthetic model build | https://github.com/MistaRichMan/smartbankAI-ml/pull/2 | https://github.com/Infinity-AI-Africa-Limited/smartbankAI-ml/pull/3 | Review reproducible synthetic data, agent artefacts, safety controls, and serving integration |
| 3 | This handoff | `manus/claude-production-handoff` | mirrored branch after review | Use as the production-hardening specification and review checklist |

## 4. ML service architecture

```text
Customer Web / Mobile / Operator Portal
              │
              ▼
SmartBank Application API (Node.js/tRPC)
  • RBAC and tenant isolation
  • consent / workflow state
  • AI Gateway: minimise payload, sign request, timeout, circuit breaker
  • immutable decision audit
              │
              ▼
ML Orchestrator (FastAPI)
  • contract version validation
  • route selection and circuit breakers
  • service-to-service token validation
              │
     ┌────────┴────────────────────────────────────────┐
     ▼         ▼           ▼            ▼               ▼
 Fraud      Credit        AML      Personalization  Predictive
               ▼           ▼            ▼               ▼
          Conversational • Aggregation • Smart Dashboard
```

### Contract and audit boundary

The platform contract is `contracts/ml-orchestrator.v1.openapi.yaml` in the application repository. The shared typed platform contract is `shared/ml-contract.ts`. The ML side mirrors the OpenAPI contract and has Pydantic schemas under `shared/schemas/orchestrator_v1.py` on the ML contract branch.

All ML calls must include a correlation ID, contract version, request purpose, and a minimal scoped payload. Results must be retained as append-only advisory audit events containing the outcome, confidence, explanation, model version, contract version, latency, and human review status. Never put raw BVN, NIN, full card PAN, passwords, or account balances into an agent prompt, model feature log, or developer log.

## 5. Eight-agent synthetic build

The model build is explicitly **development-only**. It produces reproducible synthetic Nigerian banking data and artefacts so teams can exercise pipelines, contracts, and governance controls before a bank provides authorised UAT data.

| Agent | Development artefact | Synthetic coverage | Production replacement required |
|---|---|---|---|
| Fraud Detection | LightGBM classifier, anomaly overlay, threshold and feature-contribution artefacts | transaction velocity, amount, channel, device/time risk and synthetic fraud labels | confirmed fraud/chargeback labels; transaction graph features; independent fairness and calibration review |
| Credit Risk | Explainable scorecard and challenger model | synthetic application, affordability, repayment/default outcomes | approved credit-policy data, bureau/alternative-data permissions, PD calibration, policy sign-off |
| Compliance & AML | rules, typology configuration, graph-anomaly artefact, SAR-draft inputs | structuring, layering, mule/smurfing and transaction network fixtures | approved typologies, sanctions/PEP feeds, MLRO approval, SAR workflow integration |
| Personalization | recommendation, next-best-action and segmentation artefacts | synthetic product interactions and customer segments | consented behavioural data, opt-out controls, treatment-effect evaluation |
| Predictive Analytics | churn, cash-flow and volume baselines | synthetic event and transaction time series | bank time series, back-testing, drift monitoring, business-owner thresholds |
| Conversational | synthetic knowledge corpus, retrieval baseline and safety test corpus | product FAQ, regulated-language and unsafe-action prompts | approved bank knowledge base, retrieval ACLs, prompt-injection tests, red-team assessment |
| Data Aggregation | Finacle/NIP-like source fixtures and entity-resolution evaluation set | schema normalisation and duplicate/entity linkage | source-specific mapping, reconciliation rules, lineage and exception workflow |
| Smart Dashboard | segmentation and deterministic narrative/insight artefacts | synthetic customer/portfolio KPIs | governed KPI definitions, role/tenant access checks, report reconciliation |

### Synthetic-data controls

The synthetic build generator is `scripts/generate_synthetic_data.py`; the quality validator is `scripts/validate_synthetic_data.py`; and the end-to-end development build entry point is `scripts/build_synthetic_models.sh`. Generated datasets and serialised artefacts are intentionally excluded from Git. The model/data cards and the training runbook are in `docs/` on the synthetic-model branch.

Synthetic performance is **not a production performance claim**. Do not present its metrics to a bank risk committee as validation evidence. Synthetic data is useful for proving reproducibility, service wiring, test coverage, and artefact lifecycle—not for demonstrating real-world fraud, credit, AML, or fairness performance.

## 6. Security hardening completed

The dependency-hardening branch remediates the audited Python dependency manifests and adds a CI gate.

| Control | Implemented change |
|---|---|
| Vulnerable retrieval stack | Removed unused LangChain, langchain-community, sentence-transformers, and Chroma dependencies from the conversational service; the service uses an explicit no-retrieval boundary until an approved retriever is deployed |
| Framework baseline | Updated FastAPI, Starlette, Prometheus instrumentation, and python-multipart to tested patched pins |
| ML/XML packages | Updated LightGBM and lxml pins across relevant agents |
| Reproducible audit | Added `scripts/audit_dependencies.sh`, which audits all ten agent/base/orchestrator requirement manifests with `pip-audit` |
| Regression protection | Added dependency pin tests and CI enforcement; validated audit, Ruff linting, Python compilation, and pytest |

The CI configuration currently resides at `ci/deploy.yml`. Before the first production release, Claude should move or duplicate it under `.github/workflows/` and confirm the selected GitHub Actions permissions, container registry naming, secret scopes, and environment protection rules.

## 7. Required production-hardening work

### P0 — must complete before bank UAT

1. **Identity, access, and tenancy:** replace prototype token handling with bank-approved mTLS or workload identity, short-lived service tokens, key rotation, tenant-scoped authorization, and service policy enforcement.
2. **Secrets:** move all credentials to the bank secret manager; remove every development fallback; rotate tokens used during this prototype handoff.
3. **Data protection:** define a field-level data contract; implement payload tokenisation/pseudonymisation, structured-log redaction, encrypted storage, retention, deletion, and legal-hold controls.
4. **Network controls:** deploy private-only services with default-deny network policies, egress allowlists, WAF/reverse-proxy limits, trusted-host enforcement, TLS/mTLS, and observability that does not leak customer data.
5. **Model governance:** add model registry, signed artefacts, immutable training/validation lineage, model cards approved by model risk, reproducible environment lock files, rollback, kill switch, drift monitoring, and retraining approvals.
6. **Human approval:** ensure platform workflows cannot execute transfers, blocks, loan approvals, SAR filings, or customer-impacting changes from an agent result alone.
7. **Testing:** add end-to-end integration tests for every agent route, negative authorization tests, contract tests, privacy tests, load/chaos tests, SAST/DAST/SCA, SBOM generation, container scanning, and UAT test evidence.

### P1 — should complete during UAT

1. Replace synthetic assets with bank-approved de-identified or controlled UAT data.
2. Calibrate thresholds by tenant and channel; complete bias, explainability, stability, and false-positive/false-negative analysis.
3. Integrate core banking, NIP, KYC/AML, bureau, and bank notification systems through contract-tested adapters.
4. Implement MLRO queues, fraud-investigator review queues, decision overrides, reason capture, exception escalation, and regulator-report templates.
5. Establish SLOs, runbooks, alert ownership, incident response, operational dashboards, backup/restore tests, and disaster recovery exercises.

## 8. Deployment target

The preferred first deployment is **private staging**, not public SaaS production. Run the orchestrator and agent services in the bank’s private cloud, on-premise Kubernetes, or an approved hybrid segment. The application gateway must be the sole permitted caller of the orchestrator. Model artefacts should be fetched at deployment from protected object storage or a model registry and mounted read-only.

For local development, the repository includes Docker Compose scaffolding. Use it only with synthetic assets. For staging, implement image signing, vulnerability scanning, admission policy, namespace isolation, resource limits, horizontal scaling, persistent audit storage, database backup, and environment-specific secret injection. Do not use local compose credentials, seeded data, or generated model files in a production image.

## 9. Suggested Claude Code review sequence

```bash
# 1. Inspect the dependency baseline first.
git checkout manus/dependency-hardening
./scripts/audit_dependencies.sh
ruff check agents orchestrator shared tests
pytest tests/unit -q

# 2. Review the synthetic model branch separately.
git checkout manus/synthetic-agent-models
./scripts/build_synthetic_models.sh
python3 scripts/validate_synthetic_data.py
pytest tests/unit -q

# 3. Inspect application integration and contract compatibility.
cd ../smartbankai-platform
pnpm install --frozen-lockfile
pnpm check
pnpm test
pnpm contract:check
```

Then perform the P0 review above, make production hardening changes on new branches, and merge only through protected PRs after independent approval from engineering, security, model risk, compliance/MLRO, and the bank’s designated UAT owner.

## 10. Definition of done for production readiness

A production release is not ready until all of the following are evidenced: approved architecture/security review; completed threat model; bank-approved data processing agreement; tenant isolation; mTLS and secret rotation; zero critical/high dependency and container findings; approved model validation report; human-review controls tested; UAT sign-off; CBN/compliance review where applicable; complete monitoring, support runbooks, rollback, backup/restore, and DR exercises.

## 11. Non-goals and explicit limitations

- The synthetic models are not trained on real customer, fraud, credit, or AML outcomes.
- The platform is not a core banking replacement and must not become a system of record for balances or transaction posting without bank-approved integration controls.
- No autonomous decision may be represented as a credit decision, fraud block, AML filing, or money movement authority.
- The current conversational service intentionally has no dynamic/untrusted retrieval path; do not reintroduce one without a documented retrieval ACL, approved source ingest, prompt-injection controls, and security assessment.

## References

[1]: https://github.com/MistaRichMan/smartbankAI-ml/pull/1 "Dependency hardening pull request"
[2]: https://github.com/MistaRichMan/smartbankAI-ml/pull/2 "Synthetic advisory model build pull request"
[3]: https://github.com/Infinity-AI-Africa-Limited/smartbankai-platform "SmartBank AI application platform"
