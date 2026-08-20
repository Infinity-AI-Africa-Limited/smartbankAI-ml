"""Validate the SmartBank AI synthetic-data build before model training."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate(root: Path) -> dict[str, int | float]:
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    require(manifest.get("synthetic_only") is True, "Manifest must be marked synthetic_only")
    require(manifest.get("marker") == "SYNTHETIC_DEVELOPMENT_DATA_ONLY", "Synthetic marker mismatch")
    expected = {
        "customers.csv", "transactions.csv", "loan_applications.csv", "aml_transactions.csv",
        "product_interactions.csv", "customer_activity.csv", "daily_balances.csv", "platform_daily_volume.csv",
    }
    actual = {entry["file"] for entry in manifest["datasets"]}
    require(expected.issubset(actual), f"Missing expected datasets: {expected - actual}")
    customers = pd.read_csv(root / "customers.csv")
    transactions = pd.read_csv(root / "transactions.csv")
    loans = pd.read_csv(root / "loan_applications.csv")
    aml = pd.read_csv(root / "aml_transactions.csv")
    interactions = pd.read_csv(root / "product_interactions.csv")
    balances = pd.read_csv(root / "daily_balances.csv")
    volumes = pd.read_csv(root / "platform_daily_volume.csv")
    require(len(customers) >= 1_000, "Customer dataset is too small")
    require(len(transactions) >= 10_000, "Transaction dataset is too small")
    require(transactions["is_fraud"].nunique() == 2, "Fraud labels require both classes")
    require(loans["outcome"].nunique() == 2, "Loan outcomes require both classes")
    require(set(["structuring", "smurfing"]).issubset(set(aml["typology_label"])), "AML typologies missing")
    require(interactions["product"].nunique() >= 5, "Recommendation product coverage is insufficient")
    require(balances["customer_id"].nunique() >= 100, "Cash-flow customer coverage is insufficient")
    require(len(volumes) >= 365, "Volume history requires at least one year")
    all_text = "\n".join(customers.astype(str).agg(" ".join, axis=1).head(100).tolist())
    require(not re.search(r"(?<!\d)\d{11}(?!\d)", all_text), "Potential real-style 11-digit identifier detected")
    for path in [
        root / "aggregation_fixtures" / "finacle_transactions.csv",
        root / "aggregation_fixtures" / "mobile_transactions.json",
        root / "aggregation_fixtures" / "nip_settlement.xml",
        root / "conversational" / "knowledge_base.jsonl",
        root / "conversational" / "retrieval_evaluation.csv",
    ]:
        require(path.exists() and path.stat().st_size > 0, f"Missing or empty fixture: {path}")
    report = {
        "customers": len(customers), "transactions": len(transactions), "fraud_rate": round(float(transactions.is_fraud.mean()), 5),
        "loans": len(loans), "loan_repayment_rate": round(float(loans.outcome.mean()), 5),
        "aml_transactions": len(aml), "confirmed_sar_rate": round(float(aml.confirmed_sar.mean()), 5),
        "interactions": len(interactions), "balance_observations": len(balances), "volume_days": len(volumes),
    }
    (root / "data_quality_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate SmartBank synthetic data")
    parser.add_argument("--data-dir", default="data/synthetic")
    args = parser.parse_args()
    result = validate(Path(args.data_dir))
    print(json.dumps(result, indent=2))
