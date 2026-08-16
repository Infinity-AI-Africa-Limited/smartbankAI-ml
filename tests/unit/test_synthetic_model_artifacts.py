"""Loadability and minimal-inference checks for all generated synthetic agent artefacts."""

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "synthetic"


def test_fraud_artifact_scores_and_exports_feature_contributions():
    with (ROOT / "agents/fraud_detection/models/fraud_lgbm.pkl").open("rb") as handle:
        model = pickle.load(handle)
    transaction = pd.read_csv(DATA / "transactions.csv").head(1)
    values = transaction.iloc[0]
    features = np.array([[values.amount_ngn, 0, values.hour_of_day, values.day_of_week, 0.0, 0, 0, values.sender_txn_count_1h, values.sender_30d_avg_amount]])
    assert 0 <= float(model.predict_proba(features)[0][1]) <= 1
    assert model.booster_.predict(features, pred_contrib=True).shape[1] == 10
    with (ROOT / "agents/fraud_detection/models/fraud_isolation_forest.pkl").open("rb") as handle:
        anomaly = pickle.load(handle)
    named_features = pd.DataFrame(features, columns=anomaly["feature_columns"])
    assert anomaly["model"].score_samples(named_features).shape == (1,)


def test_credit_scorecard_and_challenger_are_service_loadable():
    with (ROOT / "agents/credit_risk/models/credit_scorecard.pkl").open("rb") as handle:
        scorecard = pickle.load(handle)
    loan = pd.read_csv(DATA / "loan_applications.csv").head(1).iloc[0]
    features = pd.DataFrame([{
        "monthly_income_ngn": loan.monthly_income_ngn, "loan_amount_requested_ngn": loan.loan_amount_requested_ngn,
        "loan_tenure_months": loan.loan_tenure_months, "existing_monthly_obligations_ngn": loan.existing_monthly_obligations_ngn,
        "repayment_history_score": loan.repayment_history_score, "account_age_months": loan.account_age_months,
        "avg_monthly_balance_ngn": loan.avg_monthly_balance_ngn, "employment_type": loan.employment_type, "bvn_verified": loan.bvn_verified,
    }])
    assert 0 <= float(scorecard.probability_of_default(features)[0]) <= 1


def test_remaining_agent_artefacts_are_present_and_explicitly_synthetic():
    model_paths = [
        ROOT / "agents/aml_compliance/models/aml_graph_isolation.pkl",
        ROOT / "agents/personalization/models/nba_lgbm.pkl",
        ROOT / "agents/personalization/models/kmeans_segments.pkl",
        ROOT / "agents/predictive_analytics/models/churn_lgbm.pkl",
        ROOT / "agents/predictive_analytics/models/cashflow_ridge.pkl",
        ROOT / "agents/predictive_analytics/models/volume_autoreg_ridge.pkl",
        ROOT / "agents/conversational_ai/models/tfidf_retriever.pkl",
        ROOT / "agents/smart_dashboard/models/dashboard_kmeans.pkl",
        ROOT / "agents/data_aggregation/models/entity_match_logreg.pkl",
    ]
    for path in model_paths:
        assert path.exists(), path
        with path.open("rb") as handle:
            assert pickle.load(handle) is not None
    for report in ROOT.glob("agents/*/models/*report*.json"):
        payload = json.loads(report.read_text(encoding="utf-8"))
        assert payload.get("synthetic_only") is True, report
    conversational_report = json.loads((ROOT / "agents/conversational_ai/models/retrieval_evaluation_report.json").read_text(encoding="utf-8"))
    assert conversational_report["safety_pass_rate"] == 1.0
    assert (ROOT / "agents/data_aggregation/models/entity_resolution_evaluation.csv").exists()
