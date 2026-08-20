# SmartBank AI Synthetic Model Build Specification

## Purpose and scope

This build creates reproducible **synthetic** Nigerian banking data, model artefacts, evaluation reports, and model cards for SmartBank AI’s eight advisory agents. The artefacts are intended for development, contract testing, user-interface demonstrations, and end-to-end service validation. They are **not valid for production decisioning, CBN/NFIU filing, customer action, or model-performance claims**.

All generated records use deterministic pseudonyms and fictional identifiers. No real BVN, NIN, account number, customer, merchant, CBN circular, regulatory report, or bank data is generated or represented as real.

## Agent specifications

| Agent | Synthetic training assets | Initial model / pipeline | Primary offline measures | Serving posture |
|---|---|---|---|---|
| Fraud Detection | 120,000 labelled transactions across channels, device/location risk patterns, account aggregates | LightGBM classifier plus Isolation Forest anomaly overlay | PR-AUC, ROC-AUC, precision and recall at review threshold | Fraud referral only; never blocks a transaction |
| Credit Risk | 25,000 completed synthetic loan observations with 18-month outcomes | Explainable logistic scorecard and LightGBM challenger | ROC-AUC, Brier score, calibration error, score stability | Approve/refer/decline is a recommendation requiring officer sign-off |
| AML / Compliance | 50,000 synthetic transaction-network edges and SAR/typology labels | Deterministic typology rules plus graph anomaly score | Typology detection recall, precision, graph anomaly separation | Draft alert/SAR evidence only; an AML officer decides and files |
| Personalization | Customer-product interactions, engagement events, product adoption labels | Item similarity recommender, next-best-action classifier, K-means segments | Recall@3, top-1 accuracy, silhouette score | Ranked suggestions only; no automatic offer or enrolment |
| Predictive Analytics | Daily synthetic balances, activity snapshots, and platform daily volumes | Ridge cash-flow forecast, churn classifier, ARIMA volume forecast | MAE/MAPE, ROC-AUC, calibration error | Forecasts and risk indicators only |
| Conversational AI | Synthetic product/FAQ/policy corpus and grounded-QA evaluation set | Retrieval corpus, TF-IDF baseline evaluator, RAG-ready documents | Recall@3 and citation coverage | Informational answers with source references; no instructions to execute banking actions |
| Smart Dashboard | Reuses synthetic customer, transaction, and churn data | K-means segments plus deterministic template NLG | Silhouette score and deterministic snapshot tests | Management insight drafts only |
| Data Aggregation | Finacle CSV, mobile JSON, NIP XML fixture sets plus duplicate-link labels | Canonical normalisation and entity-match classifier | Normalisation pass rate, precision/recall/F1 for duplicate linkage | Staging-quality output; no source-of-record updates without reconciliation |

## Dataset provenance and reproducibility

Each generated data file must be accompanied by a JSON manifest containing: generator version, fixed random seed, generation timestamp, row count, schema, label definition, and a `synthetic_only: true` marker. Data is written under `data/synthetic/` and never placed in production volumes or model registries without an explicit real-data validation process.

## Evaluation gates

The synthetic model build uses acceptance gates that check the pipeline rather than claim production validity. Each trainer must produce an evaluation JSON, model card, feature/field metadata, and a serialised artefact. The test suite verifies that all expected artefacts are present and that every returned agent result has `human_review_required: true`.

## Real-data replacement requirements

Before any bank deployment, an independent model-risk function must replace synthetic data with approved, documented, privacy-governed historical data; reproduce training; validate discrimination, calibration, fairness, stability, and drift controls; set bank-approved decision thresholds; and sign off the relevant Human-in-the-Loop policy.
