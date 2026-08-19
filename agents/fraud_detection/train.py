"""
Fraud Detection Agent — Model Training Script
Usage: python train.py --data-path /data/transactions.csv --output-dir ./models
"""
import argparse
import pickle
import logging
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, precision_recall_curve, roc_auc_score, precision_score, recall_score
from sklearn.ensemble import IsolationForest
import lightgbm as lgb
import optuna

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from shared.training import ensure_synthetic, write_model_card  # noqa: E402

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

FEATURE_COLS = [
    "amount_ngn", "channel_enc", "hour_of_day", "day_of_week",
    "amount_deviation", "is_high_risk_hour", "is_weekend",
    "velocity_1h", "sender_30d_avg"
]
TARGET_COL = "is_fraud"
CHANNEL_MAP = {"mobile": 0, "web": 1, "ussd": 2, "atm": 3, "pos": 4}


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["channel_enc"] = df["channel"].map(CHANNEL_MAP).fillna(-1).astype(int)
    df["amount_deviation"] = (
        (df["amount_ngn"] - df["sender_30d_avg_amount"]) /
        (df["sender_30d_avg_amount"] + 1)
    ).fillna(0)
    df["is_high_risk_hour"] = df["hour_of_day"].apply(lambda h: 1 if h < 5 else 0)
    df["is_weekend"] = df["day_of_week"].apply(lambda d: 1 if d >= 5 else 0)
    df["velocity_1h"] = df["sender_txn_count_1h"].fillna(0)
    df["sender_30d_avg"] = df["sender_30d_avg_amount"].fillna(0)
    return df


def objective(trial, X_train, y_train, X_val, y_val):
    params = {
        "objective": "binary",
        "metric": "auc",
        "verbosity": -1,
        "boosting_type": "gbdt",
        "num_leaves": trial.suggest_int("num_leaves", 20, 100),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 50),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, 20.0),
    }
    model = lgb.LGBMClassifier(**params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
    preds = model.predict_proba(X_val)[:, 1]
    return roc_auc_score(y_val, preds)


def train(data_path: str, output_dir: str, n_trials: int = 30):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info("Loading data from %s", data_path)
    df = pd.read_csv(data_path)
    ensure_synthetic(df, "fraud transaction data")
    df = engineer_features(df)

    X = df[FEATURE_COLS]
    y = df[TARGET_COL]

    logger.info("Class distribution: %s", y.value_counts().to_dict())
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.15, random_state=42, stratify=y_train
    )

    logger.info("Running Optuna hyperparameter search (%d trials)...", n_trials)
    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda trial: objective(trial, X_train, y_train, X_val, y_val),
        n_trials=n_trials,
        show_progress_bar=True,
    )

    best_params = study.best_params
    best_params.update({"objective": "binary", "metric": "auc", "verbosity": -1})
    logger.info("Best params: %s", best_params)

    final_model = lgb.LGBMClassifier(**best_params)
    final_model.fit(X_train, y_train)

    validation_prob = final_model.predict_proba(X_val)[:, 1]
    precision_curve, recall_curve, threshold_curve = precision_recall_curve(y_val, validation_prob)
    valid = np.where(recall_curve[:-1] >= 0.75)[0]
    selected_index = int(valid[np.argmax(precision_curve[:-1][valid])]) if len(valid) else int(np.argmax(2 * precision_curve[:-1] * recall_curve[:-1] / (precision_curve[:-1] + recall_curve[:-1] + 1e-9)))
    review_threshold = float(threshold_curve[selected_index])

    y_pred_prob = final_model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_prob >= review_threshold).astype(int)

    auc = roc_auc_score(y_test, y_pred_prob)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    logger.info("Test AUC: %.4f | Precision: %.4f | Recall: %.4f", auc, precision, recall)
    logger.info("\n%s", classification_report(y_test, y_pred))

    # Save model
    model_path = output_path / "fraud_lgbm.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(final_model, f)
    logger.info("Model saved to %s", model_path)

    anomaly_model = IsolationForest(n_estimators=200, contamination=float(y_train.mean()), random_state=42)
    anomaly_model.fit(X_train[y_train == 0])
    anomaly_scores = -anomaly_model.score_samples(X_test)
    anomaly_auc = roc_auc_score(y_test, anomaly_scores)
    with open(output_path / "fraud_isolation_forest.pkl", "wb") as f:
        pickle.dump({"model": anomaly_model, "feature_columns": FEATURE_COLS}, f)

    # Export TreeSHAP-compatible per-feature contributions from LightGBM directly.
    # This avoids importing the external SHAP plotting stack in CPU-only environments.
    contributions = final_model.booster_.predict(X_test.head(25), pred_contrib=True)
    contribution_rows = []
    for row_index, values in enumerate(contributions):
        contribution_rows.append({
            "row": int(row_index),
            "feature_contributions": {feature: round(float(value), 6) for feature, value in zip(FEATURE_COLS, values[:-1], strict=False)},
            "expected_value": round(float(values[-1]), 6),
        })
    with open(output_path / "fraud_tree_shap_samples.json", "w") as f:
        json.dump(contribution_rows, f, indent=2)

    # Save evaluation report
    report = {
        "auc_roc": round(auc, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "review_threshold": round(review_threshold, 6),
        "anomaly_auc_roc": round(float(anomaly_auc), 4),
        "best_params": best_params,
        "feature_importance": dict(zip(FEATURE_COLS, final_model.feature_importances_.tolist(), strict=False)),
        "synthetic_only": True,
    }
    with open(output_path / "evaluation_report.json", "w") as f:
        json.dump(report, f, indent=2)

    write_model_card(
        output_path, "Fraud Detection", "LightGBM classifier with LightGBM TreeSHAP contribution export", "synthetic-1.0.0",
        data_path, FEATURE_COLS, report,
        ["Synthetic fraud labels reflect generator assumptions, not investigated case outcomes.", "The selected threshold only prioritises a human review queue and must not block, decline, or reverse a transaction.", "Independent validation, calibration, bias testing, and drift monitoring are required with bank-approved data."],
        "Prioritise transactions for fraud-operations review with interpretable feature contributions.",
    )

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--output-dir", default="./models")
    parser.add_argument("--n-trials", type=int, default=30)
    args = parser.parse_args()
    train(args.data_path, args.output_dir, args.n_trials)
