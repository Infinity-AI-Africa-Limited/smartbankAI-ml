"""Train development-only dashboard segments and deterministic insight templates."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from shared.training import ensure_output_dir, ensure_synthetic, write_json, write_model_card  # noqa: E402


FEATURES = ["monthly_income_ngn", "avg_monthly_balance_ngn", "account_age_months", "products_held_count", "days_since_last_transaction", "monthly_transaction_count_trend", "complaint_count_12m"]


def render_insight(metrics: dict[str, float | int]) -> str:
    volume_direction = "up" if metrics["volume_change_pct"] >= 0 else "down"
    return (
        f"Synthetic development snapshot: transaction volume is {volume_direction} {abs(metrics['volume_change_pct']):.1f}% versus the prior period. "
        f"{int(metrics['high_churn_customers'])} customers meet the synthetic high-churn threshold and require relationship-manager review. "
        f"{int(metrics['fraud_alerts'])} transaction alerts are advisory signals only and remain pending human investigation."
    )


def train(customers_path: str, activity_path: str, transactions_path: str, output_dir: str) -> dict[str, float | int | bool]:
    customers = pd.read_csv(customers_path)
    activity = pd.read_csv(activity_path)
    transactions = pd.read_csv(transactions_path)
    for name, frame in [("customers", customers), ("activity", activity), ("transactions", transactions)]:
        ensure_synthetic(frame, name)
    frame = customers.merge(activity[["customer_id", "days_since_last_transaction", "monthly_transaction_count_trend", "complaint_count_12m", "churn_probability_synthetic"]], on="customer_id")
    scaler = StandardScaler()
    values = scaler.fit_transform(frame[FEATURES])
    model = KMeans(n_clusters=5, n_init=20, random_state=42)
    labels = model.fit_predict(values)
    frame["segment_id"] = labels
    profiles = frame.groupby("segment_id")[FEATURES + ["churn_probability_synthetic"]].mean().round(3).reset_index()
    recent = transactions.sort_values("timestamp_utc").tail(max(1, len(transactions) // 10))
    previous = transactions.sort_values("timestamp_utc").iloc[-max(2, len(transactions) // 5):-max(1, len(transactions) // 10)]
    volume_change = (recent.amount_ngn.sum() / max(previous.amount_ngn.sum(), 1) - 1) * 100
    dashboard_metrics = {
        "volume_change_pct": float(volume_change),
        "high_churn_customers": int((frame.churn_probability_synthetic >= 0.5).sum()),
        "fraud_alerts": int(transactions.is_fraud.sum()),
    }
    metrics = {"segment_silhouette": round(float(silhouette_score(values, labels)), 4), "customer_count": int(len(frame)), "synthetic_only": True}
    output = ensure_output_dir(output_dir)
    with (output / "dashboard_kmeans.pkl").open("wb") as handle:
        pickle.dump({"model": model, "scaler": scaler, "feature_columns": FEATURES}, handle)
    frame[["customer_id", "segment_id", "churn_probability_synthetic"]].to_csv(output / "customer_dashboard_segments.csv", index=False)
    profiles.to_csv(output / "segment_profiles.csv", index=False)
    insight = {"metrics": dashboard_metrics, "insight": render_insight(dashboard_metrics), "synthetic_only": True}
    (output / "insight_snapshot.json").write_text(json.dumps(insight, indent=2), encoding="utf-8")
    write_json(output / "evaluation_report.json", metrics)
    write_model_card(
        output, "Smart Dashboard", "K-means segmentation and deterministic template insight generator", "synthetic-1.0.0",
        customers_path, FEATURES, metrics,
        ["The generated segment and metric patterns are fictional development examples.", "Insight text is deterministic and must be reconciled against approved reporting sources before use.", "Dashboard summaries do not replace management review, risk review, or regulated reporting."],
        "Help authorised managers prioritise review by summarising synthetic trends and segments.",
    )
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--customers-path", default="data/synthetic/customers.csv")
    parser.add_argument("--activity-path", default="data/synthetic/customer_activity.csv")
    parser.add_argument("--transactions-path", default="data/synthetic/transactions.csv")
    parser.add_argument("--output-dir", default="agents/smart_dashboard/models")
    args = parser.parse_args()
    print(train(args.customers_path, args.activity_path, args.transactions_path, args.output_dir))
