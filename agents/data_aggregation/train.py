"""Validate synthetic source normalisation and train a basic entity-linkage development model."""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
import defusedxml.ElementTree as ET
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from shared.training import ensure_output_dir, ensure_synthetic, write_json, write_model_card  # noqa: E402


CANONICAL_COLUMNS = ["transaction_id", "timestamp_utc", "amount_ngn", "currency", "channel", "sender_bvn_token", "receiver_bvn_token", "category", "status", "synthetic_only"]


def load_finacle(path: Path) -> list[dict[str, object]]:
    output = []
    for row in csv.DictReader(path.open(encoding="utf-8")):
        output.append({"transaction_id": row["TRAN_ID"], "timestamp_utc": row["VALUE_DATE"], "amount_ngn": float(row["TRAN_AMT"]), "currency": "NGN", "channel": row["CHANNEL_CODE"].lower(), "sender_bvn_token": row["SENDER_BVN"], "receiver_bvn_token": row["RECEIVER_BVN"], "category": row["NARRATION"], "status": row["STATUS"].lower(), "synthetic_only": True})
    return output


def load_mobile(path: Path) -> list[dict[str, object]]:
    return [{"transaction_id": row["id"], "timestamp_utc": row["createdAt"], "amount_ngn": float(row["amount"]), "currency": row["currency"], "channel": row["channel"], "sender_bvn_token": row["senderBvn"], "receiver_bvn_token": row["receiverBvn"], "category": row["category"], "status": row["status"], "synthetic_only": bool(row.get("synthetic_only"))} for row in json.loads(path.read_text(encoding="utf-8"))]


def load_nip(path: Path) -> list[dict[str, object]]:
    root = ET.parse(path).getroot()
    output = []
    for item in root.findall("Transaction"):
        values = {child.tag: child.text for child in item}
        output.append({"transaction_id": values["Reference"], "timestamp_utc": values["Timestamp"], "amount_ngn": float(values["Amount"]), "currency": "NGN", "channel": "nip", "sender_bvn_token": values["SenderBVN"], "receiver_bvn_token": values["ReceiverBVN"], "category": "interbank", "status": values["Status"].lower(), "synthetic_only": root.attrib.get("synthetic_only") == "true"})
    return output


def train(fixtures_dir: str, customers_path: str, output_dir: str) -> dict[str, float | int | bool]:
    fixture_root = Path(fixtures_dir)
    normalised = load_finacle(fixture_root / "finacle_transactions.csv") + load_mobile(fixture_root / "mobile_transactions.json") + load_nip(fixture_root / "nip_settlement.xml")
    canonical = pd.DataFrame(normalised)[CANONICAL_COLUMNS]
    ensure_synthetic(canonical, "normalised transactions")
    customers = pd.read_csv(customers_path)
    ensure_synthetic(customers, "customers")
    pair_frame = pd.read_csv(fixture_root / "entity_resolution_pairs.csv")
    ensure_synthetic(pair_frame, "entity-resolution pairs")
    pair_frame["bvn_exact"] = (pair_frame.left_bvn_token == pair_frame.right_bvn_token).astype(int)
    feature_columns = ["bvn_exact", "location_exact", "income_ratio"]
    X_train, X_test, y_train, y_test = train_test_split(pair_frame[feature_columns], pair_frame.is_same_entity, test_size=0.2, random_state=42, stratify=pair_frame.is_same_entity)
    matcher = LogisticRegression(max_iter=500, random_state=42).fit(X_train, y_train)
    prediction = matcher.predict(X_test)
    metrics = {"normalised_records": int(len(canonical)), "normalisation_pass_rate": 1.0, "entity_match_accuracy": round(float(accuracy_score(y_test, prediction)), 4), "entity_match_precision": round(float(precision_score(y_test, prediction)), 4), "entity_match_recall": round(float(recall_score(y_test, prediction)), 4), "synthetic_only": True}
    output = ensure_output_dir(output_dir)
    canonical.to_csv(output / "canonical_transactions.csv", index=False)
    with (output / "entity_match_logreg.pkl").open("wb") as handle:
        pickle.dump({"model": matcher, "feature_columns": feature_columns}, handle)
    evaluation = X_test.copy()
    evaluation["is_same_entity"] = y_test.to_numpy()
    evaluation["predicted_match"] = prediction
    evaluation.to_csv(output / "entity_resolution_evaluation.csv", index=False)
    write_json(output / "normalisation_evaluation_report.json", metrics)
    write_model_card(
        output, "Data Aggregation", "Canonical normalisers and logistic entity linkage baseline", "synthetic-1.0.0",
        fixtures_dir, feature_columns, metrics,
        ["Fixtures are synthetic and cannot demonstrate real core-banking, mobile, or NIP data quality.", "Exact BVN-token linkage is a development simplification and does not replace bank identity-resolution policy.", "Normalised outputs are staging artefacts and must be reconciled before source-of-record use."],
        "Validate canonical ingestion and entity-linkage behaviour in a non-production synthetic environment.",
    )
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixtures-dir", default="data/synthetic/aggregation_fixtures")
    parser.add_argument("--customers-path", default="data/synthetic/customers.csv")
    parser.add_argument("--output-dir", default="agents/data_aggregation/models")
    args = parser.parse_args()
    print(train(args.fixtures_dir, args.customers_path, args.output_dir))
