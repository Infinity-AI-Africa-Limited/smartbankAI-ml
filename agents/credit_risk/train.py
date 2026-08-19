"""Train development-only synthetic credit scorecard and challenger artefacts."""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from agents.credit_risk.scorecard import CATEGORICAL_FEATURES, FEATURES, NUMERIC_FEATURES, WoeScorecard  # noqa: E402
from shared.training import ensure_output_dir, ensure_synthetic, write_json, write_model_card  # noqa: E402


def calculate_woe(values: pd.Series, defaulted: pd.Series) -> dict[str, float]:
    summary = pd.DataFrame({"bucket": values.astype(str), "defaulted": defaulted}).groupby("bucket", observed=False).defaulted.agg(["sum", "count"])
    total_bad = max(float(defaulted.sum()), 1.0)
    total_good = max(float((1 - defaulted).sum()), 1.0)
    good = summary["count"] - summary["sum"]
    return {
        str(bucket): float(np.log(((good_value + 0.5) / total_good) / ((bad_value + 0.5) / total_bad)))
        for bucket, good_value, bad_value in zip(summary.index, good, summary["sum"], strict=False)
    }


def fit_scorecard(frame: pd.DataFrame, defaulted: pd.Series) -> WoeScorecard:
    edges: dict[str, list[float]] = {}
    mappings: dict[str, dict[str, float]] = {}
    woe_features: dict[str, pd.Series] = {}
    for feature in NUMERIC_FEATURES:
        raw = pd.to_numeric(frame[feature], errors="coerce").fillna(0)
        quantiles = np.unique(np.quantile(raw, np.linspace(0, 1, 7)))
        if len(quantiles) < 3:
            quantiles = np.array([-np.inf, float(raw.median()), np.inf])
        quantiles[0], quantiles[-1] = -np.inf, np.inf
        bucket = np.digitize(raw, quantiles[1:-1], right=True)
        edges[feature] = quantiles.tolist()
        mappings[feature] = calculate_woe(pd.Series(bucket, index=frame.index), defaulted)
        woe_features[feature] = pd.Series(bucket, index=frame.index).astype(str).map(mappings[feature]).fillna(0.0)
    for feature in CATEGORICAL_FEATURES:
        bucket = frame[feature].astype(str)
        mappings[feature] = calculate_woe(bucket, defaulted)
        woe_features[feature] = bucket.map(mappings[feature]).fillna(0.0)
    encoded = pd.DataFrame(woe_features)[FEATURES]
    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression(max_iter=1_000, class_weight="balanced", random_state=42)
    model.fit(encoded, defaulted)
    return WoeScorecard(edges, mappings, model)


def train(data_path: str, output_dir: str) -> dict[str, float]:
    frame = pd.read_csv(data_path)
    ensure_synthetic(frame, "credit data")
    frame["defaulted"] = 1 - frame["outcome"].astype(int)
    train_frame, test_frame = train_test_split(frame, test_size=0.2, random_state=42, stratify=frame["defaulted"])
    scorecard = fit_scorecard(train_frame, train_frame["defaulted"])
    scorecard_pd = scorecard.probability_of_default(test_frame)
    challenger = lgb.LGBMClassifier(
        objective="binary", n_estimators=180, learning_rate=0.05, num_leaves=24,
        min_child_samples=70, subsample=0.85, colsample_bytree=0.9, random_state=42, verbosity=-1,
    )
    challenger_frame = pd.get_dummies(train_frame[FEATURES], columns=CATEGORICAL_FEATURES, dtype=float)
    challenger_test = pd.get_dummies(test_frame[FEATURES], columns=CATEGORICAL_FEATURES, dtype=float).reindex(columns=challenger_frame.columns, fill_value=0)
    challenger.fit(challenger_frame, train_frame["defaulted"])
    challenger_pd = challenger.predict_proba(challenger_test)[:, 1]
    metrics = {
        "scorecard_auc_roc": round(float(roc_auc_score(test_frame.defaulted, scorecard_pd)), 4),
        "scorecard_brier": round(float(brier_score_loss(test_frame.defaulted, scorecard_pd)), 4),
        "challenger_auc_roc": round(float(roc_auc_score(test_frame.defaulted, challenger_pd)), 4),
        "challenger_brier": round(float(brier_score_loss(test_frame.defaulted, challenger_pd)), 4),
        "train_rows": int(len(train_frame)), "test_rows": int(len(test_frame)),
        "default_rate": round(float(frame.defaulted.mean()), 4), "synthetic_only": True,
    }
    output = ensure_output_dir(output_dir)
    with (output / "credit_scorecard.pkl").open("wb") as handle:
        pickle.dump(scorecard, handle)
    with (output / "credit_lgbm.pkl").open("wb") as handle:
        pickle.dump({"model": challenger, "columns": challenger_frame.columns.tolist()}, handle)
    write_json(output / "evaluation_report.json", metrics)
    write_model_card(
        output, "Credit Risk", "WOE logistic scorecard with LightGBM challenger", "synthetic-1.0.0",
        data_path, FEATURES, metrics,
        ["Synthetic outcomes encode the generator assumptions rather than bank repayment behaviour.", "The challenger model is benchmarking-only; the scorecard is still advisory-only.", "Credit policy, affordability, fairness, calibration, and stability require independent validation on approved real data."],
        "Produce a probability-of-default and explainable referral recommendation for a qualified loan officer.",
    )
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default="data/synthetic/loan_applications.csv")
    parser.add_argument("--output-dir", default="agents/credit_risk/models")
    args = parser.parse_args()
    print(train(args.data_path, args.output_dir))
