# SmartBank AI Synthetic Data Card

## Dataset identity

| Field | Value |
|---|---|
| Name | SmartBank AI synthetic Nigerian-banking development corpus |
| Generator | `scripts/generate_synthetic_data.py` |
| Fixed seed | `20260816` |
| Provenance | Fully generated fictional records; no customer, account, BVN, NIN, transaction, merchant, or bank data is included |
| Intended use | Development, pipeline validation, UI demos, automated tests, and service contract tests |
| Prohibited use | Production decisioning, regulatory reporting, customer targeting, model-performance claims, or training a bank production model |

## Included assets

The generator produces customer, transaction, loan, AML-network, product-interaction, customer-activity, daily-balance, and platform-volume tables. It also produces Finacle-style CSV, mobile JSON, and NIP-style XML fixtures; plus a synthetic conversational knowledge corpus and retrieval evaluation set.

## Label semantics

| Dataset | Label | Meaning |
|---|---|---|
| `transactions.csv` | `is_fraud` | Random draw from an intentionally constructed development risk function; not an investigated fraud outcome |
| `loan_applications.csv` | `outcome` | Synthetic repayment/default draw from an artificial affordability and repayment-history relationship |
| `aml_transactions.csv` | `confirmed_sar`, `typology_label` | Generator-injected investigative patterns; not SARs, filings, or legal findings |
| `customer_activity.csv` | `churned_next_90_days` | Synthetic inactivity outcome based on generator assumptions |
| `product_interactions.csv` | `adopted` | Synthetic engagement/adoption event; not consent or suitability evidence |

## Quality controls

`scripts/validate_synthetic_data.py` verifies the manifest, synthetic marker, dataset coverage, two-class fraud and credit labels, AML typology coverage, balance/volume history length, no real-style 11-digit identifiers in sampled customer data, and source-fixture presence.

## Limitations

Synthetic distributions encode the developer’s assumptions and therefore can produce plausible-looking but non-generalizable performance. They omit real-world data quality issues, operational disputes, policy exceptions, customer behaviour, fraud adaptation, feature drift, protected-class impacts, and institutional controls. The generated artefacts must be discarded and retrained under approved data governance before a bank uses them beyond development.
