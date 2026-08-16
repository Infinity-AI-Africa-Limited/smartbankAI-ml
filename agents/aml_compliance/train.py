"""Build synthetic AML typology rules and a graph anomaly model for development use."""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from shared.training import ensure_output_dir, ensure_synthetic, write_json, write_model_card  # noqa: E402


RULE_CONFIG = {
    "synthetic_structuring_threshold_ngn": 1_000_000,
    "structuring_window_hours": 24,
    "minimum_structuring_count": 4,
    "smurfing_window_days": 7,
    "minimum_distinct_senders": 5,
    "note": "Development-only illustrative thresholds; bank policy must supply production thresholds.",
}


def node_features(frame: pd.DataFrame) -> pd.DataFrame:
    graph = nx.DiGraph()
    for row in frame.itertuples(index=False):
        graph.add_edge(row.sender_id, row.receiver_id, amount=float(row.amount_ngn))
    outgoing = frame.groupby("sender_id").agg(outgoing_count=("transaction_id", "count"), outgoing_amount=("amount_ngn", "sum"), outgoing_unique=("receiver_id", "nunique"))
    incoming = frame.groupby("receiver_id").agg(incoming_count=("transaction_id", "count"), incoming_amount=("amount_ngn", "sum"), incoming_unique=("sender_id", "nunique"))
    nodes = sorted(set(frame.sender_id).union(frame.receiver_id))
    result = pd.DataFrame(index=nodes)
    result = result.join(outgoing).join(incoming).fillna(0)
    result["in_degree"] = [graph.in_degree(node) for node in result.index]
    result["out_degree"] = [graph.out_degree(node) for node in result.index]
    result["reciprocity_proxy"] = np.minimum(result["incoming_count"], result["outgoing_count"]) / (np.maximum(result["incoming_count"], result["outgoing_count"]) + 1)
    return result.reset_index(names="entity_id")


def train(data_path: str, output_dir: str) -> dict[str, float | int | bool]:
    frame = pd.read_csv(data_path)
    ensure_synthetic(frame, "AML transaction data")
    frame["timestamp_utc"] = pd.to_datetime(frame["timestamp_utc"], utc=True)
    features = node_features(frame)
    suspicious_senders = set(frame.loc[frame.confirmed_sar == 1, "sender_id"])
    features["suspicious_label"] = features.entity_id.isin(suspicious_senders).astype(int)
    feature_cols = [column for column in features.columns if column not in {"entity_id", "suspicious_label"}]
    model = IsolationForest(n_estimators=200, contamination=0.04, random_state=42)
    model.fit(features[feature_cols])
    anomaly = -model.score_samples(features[feature_cols])
    metrics = {
        "node_count": int(len(features)), "edge_count": int(len(frame)),
        "graph_anomaly_auc_roc": round(float(roc_auc_score(features.suspicious_label, anomaly)), 4),
        "graph_anomaly_average_precision": round(float(average_precision_score(features.suspicious_label, anomaly)), 4),
        "structuring_rows": int((frame.typology_label == "structuring").sum()),
        "smurfing_rows": int((frame.typology_label == "smurfing").sum()),
        "synthetic_only": True,
    }
    output = ensure_output_dir(output_dir)
    with (output / "aml_graph_isolation.pkl").open("wb") as handle:
        pickle.dump({"model": model, "feature_columns": feature_cols}, handle)
    write_json(output / "aml_rule_config.json", RULE_CONFIG)
    write_json(output / "evaluation_report.json", metrics)
    features.assign(anomaly_score=np.round(anomaly, 6)).sort_values("anomaly_score", ascending=False).head(200).to_csv(output / "graph_entity_scores.csv", index=False)
    write_model_card(
        output, "AML / Compliance", "Deterministic typology rules and Isolation Forest graph anomaly detector", "synthetic-1.0.0",
        data_path, feature_cols, metrics,
        ["Synthetic typology labels are generator patterns, not confirmed AML investigations.", "Rules are illustrative and must be replaced with bank policy, legal, and compliance-approved thresholds.", "The graph score is an investigation prioritisation signal, not a SAR filing or transaction action."],
        "Prioritise possible transaction-network anomalies and generate investigation-ready evidence for a compliance officer.",
    )
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default="data/synthetic/aml_transactions.csv")
    parser.add_argument("--output-dir", default="agents/aml_compliance/models")
    args = parser.parse_args()
    print(train(args.data_path, args.output_dir))
