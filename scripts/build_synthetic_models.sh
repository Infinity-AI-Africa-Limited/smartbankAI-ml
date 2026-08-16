#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

DATA_DIR="${1:-data/synthetic}"

python3 scripts/generate_synthetic_data.py --output-dir "$DATA_DIR" --seed 20260816
python3 scripts/validate_synthetic_data.py --data-dir "$DATA_DIR"
python3 agents/fraud_detection/train.py --data-path "$DATA_DIR/transactions.csv" --output-dir agents/fraud_detection/models --n-trials 8
python3 agents/credit_risk/train.py --data-path "$DATA_DIR/loan_applications.csv" --output-dir agents/credit_risk/models
python3 agents/aml_compliance/train.py --data-path "$DATA_DIR/aml_transactions.csv" --output-dir agents/aml_compliance/models
python3 agents/personalization/train.py --interactions-path "$DATA_DIR/product_interactions.csv" --customers-path "$DATA_DIR/customers.csv" --output-dir agents/personalization/models
python3 agents/predictive_analytics/train.py --activity-path "$DATA_DIR/customer_activity.csv" --balances-path "$DATA_DIR/daily_balances.csv" --volume-path "$DATA_DIR/platform_daily_volume.csv" --output-dir agents/predictive_analytics/models
python3 agents/conversational_ai/build_knowledge.py --knowledge-path "$DATA_DIR/conversational/knowledge_base.jsonl" --evaluation-path "$DATA_DIR/conversational/retrieval_evaluation.csv" --output-dir agents/conversational_ai/models
python3 agents/smart_dashboard/train.py --customers-path "$DATA_DIR/customers.csv" --activity-path "$DATA_DIR/customer_activity.csv" --transactions-path "$DATA_DIR/transactions.csv" --output-dir agents/smart_dashboard/models
python3 agents/data_aggregation/train.py --fixtures-dir "$DATA_DIR/aggregation_fixtures" --customers-path "$DATA_DIR/customers.csv" --output-dir agents/data_aggregation/models

echo "Synthetic SmartBank AI model build complete. All artefacts are development-only and human-review-required."
