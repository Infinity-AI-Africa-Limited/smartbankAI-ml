# Private-Staging and Bank-Validation Gates

## Current status

The `hardening/p0` branch is a **development and review artefact**. Its passing synthetic-data tests demonstrate that the service contracts, training paths, and safety controls can be exercised in a reproducible non-production environment. They do not establish real-bank model accuracy, calibration, fairness, regulatory efficacy, or production readiness.

The ML services must not be exposed publicly and must not receive live banking data until a design-partner bank has approved the target environment and the data-processing basis.

## Private-staging activation prerequisites

| Gate | Required evidence | Status |
|---|---|---|
| Private target | Bank-approved private Kubernetes, private-cloud, or on-premises container target with private DNS and no public agent exposure. | Blocked: target not supplied. |
| Workload authentication | Bank-approved mTLS or workload identity between platform gateway, orchestrator, and named agents; unauthorised-client rejection test. | Blocked: target identity/PKI standard not supplied. |
| Secrets | Managed runtime secret injection, short-lived credentials, rotation evidence, and no values in Git, images, logs, or browser code. | Blocked: bank secret-store interface not supplied. |
| Network isolation | Default-deny ingress/egress, named service allow rules, and negative public-connectivity evidence. | Blocked: target network design not supplied. |
| Model artefacts | Development-only artefacts built inside staging, read-only mounts, and health endpoints showing `model_loaded: true`. | Blocked: staging target not supplied. |
| Platform audit path | Authenticated v1 platform-to-orchestrator call with correlation ID, advisory response, and append-only audit evidence. | Blocked: platform gateway secrets intentionally unset until the private target exists. |

## Bank UAT and model-risk gate

> A synthetic model can demonstrate pipeline behaviour. It cannot demonstrate performance on a bank's population or satisfy independent model-risk approval.

Before any controlled UAT involving bank data, the design partner must execute a UAT agreement and data-processing agreement (DPA), approve the data classification and minimisation plan, and define retention, access, and deletion controls. Independent validation must then cover representative data sampling, calibration by tenant/channel/segment, false-positive and false-negative analysis, fairness, explainability, security/privacy review, threshold approval, and MLRO/compliance sign-off.

No production promotion, accuracy claim, credit disposition, fraud action, AML filing, or autonomous customer-impacting action is allowed before those gates are independently completed and recorded.
