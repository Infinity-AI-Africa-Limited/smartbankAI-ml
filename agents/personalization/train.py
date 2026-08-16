"""Train development-only recommendation, next-best-action, and segment artefacts."""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import accuracy_score, silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from shared.training import ensure_output_dir, ensure_synthetic, write_json, write_model_card  # noqa: E402


SEGMENT_NAMES = ["New", "Growing", "Stable", "At Risk", "High Value"]


def train(interactions_path: str, customers_path: str, output_dir: str) -> dict[str, float | int | bool]:
    interactions = pd.read_csv(interactions_path)
    customers = pd.read_csv(customers_path)
    ensure_synthetic(interactions, "product interactions")
    ensure_synthetic(customers, "customer data")
    matrix = interactions.pivot_table(index="customer_id", columns="product", values="engagement_score", aggfunc="max", fill_value=0.0)
    svd = TruncatedSVD(n_components=min(4, matrix.shape[1] - 1), random_state=42)
    svd.fit_transform(matrix)
    item_embeddings = svd.components_.T
    norms = np.linalg.norm(item_embeddings, axis=1, keepdims=True) + 1e-9
    similarity = (item_embeddings / norms) @ (item_embeddings / norms).T
    similarity_frame = pd.DataFrame(similarity, index=matrix.columns, columns=matrix.columns)

    labelled = interactions.sort_values(["customer_id", "engagement_score"], ascending=[True, False]).drop_duplicates("customer_id")
    model_frame = labelled.merge(customers[["customer_id", "age", "monthly_income_ngn", "products_held_count"]], on="customer_id", how="left")
    feature_frame = pd.get_dummies(model_frame[["age", "monthly_income_ngn", "account_age_months", "products_held_count", "channel_preference", "income_band"]], dtype=float)
    target = model_frame["product"].astype("category")
    X_train, X_test, y_train, y_test = train_test_split(feature_frame, target, test_size=0.2, random_state=42, stratify=target)
    nba = lgb.LGBMClassifier(objective="multiclass", n_estimators=170, learning_rate=0.05, num_leaves=16, random_state=42, verbosity=-1)
    nba.fit(X_train, y_train)
    nba_accuracy = accuracy_score(y_test, nba.predict(X_test))

    segment_frame = customers[["monthly_income_ngn", "account_age_months", "avg_monthly_balance_ngn", "products_held_count"]].copy()
    scaler = StandardScaler()
    scaled = scaler.fit_transform(segment_frame)
    kmeans = KMeans(n_clusters=5, n_init=20, random_state=42)
    segment_labels = kmeans.fit_predict(scaled)
    profile = customers[["customer_id"]].copy()
    profile["segment_id"] = segment_labels
    rank = profile.join(customers[["avg_monthly_balance_ngn"]]).groupby("segment_id").avg_monthly_balance_ngn.mean().sort_values().index.tolist()
    name_map = {cluster: SEGMENT_NAMES[position] for position, cluster in enumerate(rank)}
    profile["segment_name"] = profile.segment_id.map(name_map)
    metrics = {
        "customer_interactions": int(len(interactions)), "customers": int(len(matrix)),
        "svd_explained_variance": round(float(svd.explained_variance_ratio_.sum()), 4),
        "next_best_action_top1_accuracy": round(float(nba_accuracy), 4),
        "segment_silhouette": round(float(silhouette_score(scaled, segment_labels)), 4),
        "synthetic_only": True,
    }
    output = ensure_output_dir(output_dir)
    similarity_frame.to_csv(output / "product_similarity.csv")
    profile.to_csv(output / "customer_segments.csv", index=False)
    with (output / "nba_lgbm.pkl").open("wb") as handle:
        pickle.dump({"model": nba, "columns": feature_frame.columns.tolist(), "classes": nba.classes_.tolist()}, handle)
    with (output / "kmeans_segments.pkl").open("wb") as handle:
        pickle.dump({"model": kmeans, "scaler": scaler, "feature_columns": segment_frame.columns.tolist(), "segment_names": name_map}, handle)
    write_json(output / "evaluation_report.json", metrics)
    write_model_card(
        output, "Personalization", "Truncated-SVD product similarity, LightGBM next-best-action, K-means segments", "synthetic-1.0.0",
        interactions_path, feature_frame.columns.tolist(), metrics,
        ["Synthetic engagement does not represent customer preference, consent, affordability, or suitability.", "Recommendations must be filtered through product eligibility, consent, fair-treatment, and campaign controls.", "No recommendation may enrol a customer or execute an action automatically."],
        "Rank potentially relevant products and customer segments for a relationship manager or customer-controlled experience.",
    )
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--interactions-path", default="data/synthetic/product_interactions.csv")
    parser.add_argument("--customers-path", default="data/synthetic/customers.csv")
    parser.add_argument("--output-dir", default="agents/personalization/models")
    args = parser.parse_args()
    print(train(args.interactions_path, args.customers_path, args.output_dir))
